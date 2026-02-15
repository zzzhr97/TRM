import asyncio
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from .abstract import AbstractRewardManager
from .registry import register


@register("parallel_naive")
class ParallelNaiveRewardManager(AbstractRewardManager):
    """
    Drop-in replacement for NaiveRewardManager that batches compute_score calls
    using asyncio (and a fallback thread pool for synchronous reward functions).
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: Callable[..., Any] | None = None,
        reward_fn_key: str = "data_source",
        parallelism: int = 32,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.parallelism = max(1, int(parallelism))
        self._async_score = getattr(self.compute_score, "async_fn", None)
        self._shutdown_async = getattr(self.compute_score, "shutdown_async_fn", None)

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | Dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources: Dict[str, int] = {}

        payloads: List[Dict[str, Any]] = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
            extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})

            payloads.append(
                {
                    "index": i,
                    "valid_response_length": int(valid_response_length),
                    "prompt_str": prompt_str,
                    "response_str": response_str,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "extra_info": extra_info,
                }
            )

        scores = asyncio.run(self._batch_compute(payloads))

        logger = logging.getLogger("parallel_naive_reward")

        for payload, score in zip(payloads, scores, strict=True):
            i = payload["index"]
            data_source = payload["data_source"]
            reward_value, info_dict = self._normalize_score(score)
            response_len = payload["valid_response_length"]
            if response_len <= 0:
                logger.warning(
                    "[parallel_naive] sample %s has empty response; skipping reward write",
                    data_source,
                )
                continue
            reward_tensor[i, response_len - 1] = reward_value

            for key, value in info_dict.items():
                reward_extra_info[key].append(value)

            seen = already_print_data_sources.setdefault(data_source, 0)
            if seen < self.num_examine:
                already_print_data_sources[data_source] = seen + 1
                logger.info(
                    "[prompt] %s\n[response] %s\n[ground_truth] %s\n[score] %s",
                    payload["prompt_str"],
                    payload["response_str"],
                    payload["ground_truth"],
                    reward_value,
                )

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor

    async def _batch_compute(self, payloads: List[Dict[str, Any]]) -> List[Any]:
        semaphore = asyncio.Semaphore(self.parallelism)

        executor: ThreadPoolExecutor | None = None

        if self._async_score:
            async def runner(payload: Dict[str, Any]) -> Any:
                async with semaphore:
                    return await self._async_score(
                        data_source=payload["data_source"],
                        solution_str=payload["response_str"],
                        ground_truth=payload["ground_truth"],
                        extra_info=payload["extra_info"],
                    )
        else:
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=self.parallelism)

            async def runner(payload: Dict[str, Any]) -> Any:
                return await loop.run_in_executor(
                    executor,
                    lambda: self.compute_score(
                        data_source=payload["data_source"],
                        solution_str=payload["response_str"],
                        ground_truth=payload["ground_truth"],
                        extra_info=payload["extra_info"],
                    ),
                )

        tasks = [asyncio.create_task(runner(payload)) for payload in payloads]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        if executor is not None:
            executor.shutdown(wait=True)

        return list(results)

    @staticmethod
    def _normalize_score(score: Any) -> tuple[float, Dict[str, Any]]:
        if isinstance(score, dict):
            base = float(score.get("score", 0.0))
            return base, score
        if isinstance(score, (float, int)):
            return float(score), {}
        return float(score[0]), {}

    def __del__(self) -> None:
        if self._shutdown_async is None:
            return
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._shutdown_async())
        except Exception:  # pragma: no cover - best effort during GC
            pass
        finally:
            loop.close()
