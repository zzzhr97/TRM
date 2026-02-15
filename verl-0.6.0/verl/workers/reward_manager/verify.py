import asyncio
import json
import logging
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
from typing import Any, Callable, Dict, List

import torch
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.custom_math_verify import math_verify_score, extract_solution
from verl.utils.debug import marked_timer
from .abstract import AbstractRewardManager
from .registry import register


def _aggregate_group_task(args):
    """
    Helper for processing a single group in parallel.
    Returns list of (global_idx, updated_score_dict).
    """
    idxs, scores = args
    rm_raw = [s.get("rm_score_raw") for s in scores]
    verify_binaries = [s.get("verify_binary") for s in scores]
    if any(x is None for x in rm_raw):
        raise ValueError("Missing rm_score_raw in group")

    rm_norm = [float(x) for x in rm_raw]
    updated_local = []
    for local_idx, global_idx in enumerate(idxs):
        s = dict(scores[local_idx])
        verify_binary = float(verify_binaries[local_idx])
        rm_score_raw = rm_raw[local_idx]
        rm_score = rm_norm[local_idx]
        alpha, beta = 0.8, 0.2
        final_score = verify_binary * (alpha + beta * VerifyRewardManager.stable_sigmoid(rm_norm[local_idx]))

        s["rm_score"] = rm_score
        s["rm_score_raw"] = float(rm_score_raw)
        s["score"] = final_score
        updated_local.append((global_idx, s))
    return updated_local


def _math_check_task(payload: Dict[str, Any]) -> float:
    try:
        student_answer, _ = extract_solution(payload.get("response_str") or "")
        if student_answer is None:
            return 0.0
        ground_truth = payload.get("ground_truth")
        return 1.0 if math_verify_score(student_answer, ground_truth) else 0.0
    except Exception:
        return 0.0

@register("verify")
class VerifyRewardManager(AbstractRewardManager):
    """
    # [VerifyRM] reward manager that fuses remote verifier scores with optional RM shaping.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: Callable[..., Any] | None = None,
        reward_fn_key: str = "data_source",
        parallelism: int = 1024,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        # ===== verify compute_score wiring =====
        wrapped_compute = compute_score or default_compute_score
        source_tag = getattr(wrapped_compute, "_verify_rm_source", None)
        if source_tag != "remote_verifier":
            msg = (
                "[VerifyRewardManager] compute_score must come from main/remote_verifier.py; "
                f"got source tag '{source_tag}'. Ensure train.sh loads that file via custom_reward_function."
            )
            logging.error(msg)
            raise ValueError(msg)
        async_hook = getattr(wrapped_compute, "async_fn", None)
        shutdown_hook = getattr(wrapped_compute, "shutdown_async_fn", None)
        if async_hook is None or shutdown_hook is None:
            msg = "[VerifyRewardManager] compute_score missing async/shutdown hooks; remote verifier interface mismatch."
            logging.error(msg)
            raise ValueError(msg)
        self.compute_score = wrapped_compute
        self._async_score = async_hook
        self._shutdown_async = shutdown_hook
        # ===== verify compute_score wiring =====
        self.reward_fn_key = reward_fn_key
        self.parallelism = max(1, int(parallelism))
        self._logger = logging.getLogger("verify_reward")

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | Dict[str, Any]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources: Dict[str, int] = {}

        payloads: List[Dict[str, Any]] = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
            extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})

            unique_id = (
                f"{data_item.non_tensor_batch['data_source']}:"
                f"{extra_info.get('index', '')}:"
                f"{extra_info.get('split', '')}:"
                f"{extra_info.get('category') or extra_info.get('subset') or 'full'}:"
                f"{extra_info['user_prompt']}"
            )

            payloads.append(
                {
                    "index": i,
                    "valid_response_length": valid_response_length,
                    "prompt_str": self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True),
                    "response_str": self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
                    "ground_truth": data_item.non_tensor_batch["reward_model"]["ground_truth"],
                    "data_source": data_item.non_tensor_batch[self.reward_fn_key],
                    "extra_info": extra_info,
                    "unique_id": unique_id,
                }
            )

        timing_reward: Dict[str, float] = {}

        with marked_timer("math_check_all", timing_reward):
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Math verify (reward manager)"),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("math", total=len(payloads))
                flags = [0.0] * len(payloads)
                with ProcessPoolExecutor(max_workers=32) as pool:
                    future_to_i = {pool.submit(_math_check_task, p): i for i, p in enumerate(payloads)}
                    try:
                        for fut in as_completed(future_to_i, timeout=20):
                            i = future_to_i[fut]
                            try:
                                flags[i] = float(fut.result())
                            except Exception:
                                flags[i] = 0.0
                            progress.update(task, advance=1)
                    except TimeoutError:
                        pool.shutdown(wait=False, cancel_futures=False)


        for payload, flag in zip(payloads, flags, strict=True):
            extra = payload.get("extra_info") or {}
            extra["math_verify_score"] = float(flag)
            payload["extra_info"] = extra

        with marked_timer("batch_compute_all", timing_reward):
            scores = asyncio.run(self._batch_compute(payloads))

        # Group by unique_id to aggregate rollout scores and normalize RM per group
        updated_scores = self._aggregate_group_scores(payloads, scores)

        group2idxs: Dict[str, List[int]] = defaultdict(list)
        for payload in payloads:
            group2idxs[payload["unique_id"]].append(payload["index"])

        first_group_size = len(next(iter(group2idxs.values()))) if group2idxs else 0
        record_verify_k = first_group_size > 1
        if record_verify_k:
            for idxs in group2idxs.values():
                if len(idxs) != first_group_size:
                    record_verify_k = False
                    break
        verify_bins = first_group_size + 1 if record_verify_k else 0

        idx2verify_k: List[int] = [0] * len(updated_scores)
        if record_verify_k:
            for uid, idxs in group2idxs.items():
                correct_cnt = 0
                for idx in idxs:
                    vb = updated_scores[idx].get("verify_binary")
                    try:
                        correct_cnt += 1 if float(vb) >= 0.5 else 0
                    except Exception:
                        correct_cnt += 1 if bool(vb) else 0
                correct_cnt = max(0, min(first_group_size, int(correct_cnt)))
                for idx in idxs:
                    idx2verify_k[idx] = correct_cnt

        for payload, updated_score in zip(payloads, updated_scores, strict=True):
            i = payload["index"]
            data_source = payload["data_source"]

            final_score, info_dict = updated_score["score"], updated_score

            # verify component
            verify_component = info_dict['verify_score']
            passed = info_dict['verify_binary']

            # reward model component
            rm_component = info_dict["rm_score"]
            raw_rm = info_dict["rm_score_raw"]

            # flag missing student answer
            verify_missing_student_answer = bool(info_dict.get("verify_missing_student_answer"))

            # response length
            response_len = payload["valid_response_length"]
            if response_len <= 0:
                self._logger.warning(
                    "[VerifyRewardManager] sample %s has empty response; skipping reward write",
                    data_source,
                )
                continue

            # final score
            reward_tensor[i, response_len - 1] = final_score

            # logging reward components
            reward_extra_info["final_scores"].append(final_score)
            reward_extra_info["verify_scores"].append(verify_component)
            reward_extra_info["verify_binary"].append(passed)
            boxed_flag = 1.0 if not verify_missing_student_answer else 0.0
            reward_extra_info["boxed"].append(boxed_flag)
            if rm_component is None or raw_rm is None:
                raise ValueError(f"Missing RM data for sample {i} ({data_source}); info={info_dict}")
            reward_extra_info["rm_scores"].append(rm_component)
            reward_extra_info["rm_scores_raw"].append(raw_rm)
            if info_dict:
                meta = dict(info_dict)
                meta.pop("verify_missing_student_answer", None)
                reward_extra_info["verify_meta"].append(json.dumps(meta, ensure_ascii=False))

            # ===== ADDED: write verify_0..verify_N (one-hot) per sample =====
            if record_verify_k:
                k = idx2verify_k[i]
                for kk in range(verify_bins):
                    reward_extra_info[f"verify_{kk}"].append(1.0 if kk == k else 0.0)
            # ===== END ADDED: write verify_0..verify_N (one-hot) per sample =====

            seen = already_print_data_sources.setdefault(data_source, 0)
            if seen < self.num_examine:
                already_print_data_sources[data_source] = seen + 1
                self._logger.info(
                    "[prompt] %s\n[response] %s\n[ground_truth] %s\n[verify] %.4f\n[rm] %.4f\n[final] %.4f",
                    payload["prompt_str"],
                    payload["response_str"],
                    payload["ground_truth"],
                    verify_component,
                    rm_component if rm_component is not None else 1.0,
                    final_score,
                )

        # ===== ADDED: reward timing per-sample copy =====
        if timing_reward:
            for name, val in timing_reward.items():
                key = f"reward_timing/{name}"
                reward_extra_info[key] = [float(val)] * len(payloads)
        # ===== END ADDED: reward timing per-sample copy =====

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra_info)}
        return reward_tensor

    async def _batch_compute(self, payloads: List[Dict[str, Any]]) -> List[Any]:
        """
        Optimized batch compute:
        - preserves order with payloads
        - limits in-flight tasks (no task explosion)
        - robust progress update
        """

        total = len(payloads)
        if total == 0:
            return []

        # ===== concurrency controls =====
        parallelism = max(1, int(self.parallelism))
        semaphore = asyncio.Semaphore(parallelism)

        # ===== choose execution mode =====
        executor: ThreadPoolExecutor | None = None

        if self._async_score is not None:
            async def run_one(idx: int, payload: Dict[str, Any]):
                async with semaphore:
                    res = await self._async_score(
                        data_source=payload["data_source"],
                        solution_str=payload["response_str"],
                        ground_truth=payload["ground_truth"],
                        extra_info=payload["extra_info"],
                    )
                    return idx, res

        else:
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=parallelism)

            async def run_one(idx: int, payload: Dict[str, Any]):
                async with semaphore:
                    res = await loop.run_in_executor(
                        executor,
                        lambda: self.compute_score(
                            data_source=payload["data_source"],
                            solution_str=payload["response_str"],
                            ground_truth=payload["ground_truth"],
                            extra_info=payload["extra_info"],
                        ),
                    )
                    return idx, res

        # ===== progress =====
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Remote verify / RM"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            transient=True,
        )
        task_id = progress.add_task("verify", total=total)
        progress.start()

        results: List[Any] = [None] * total  # type: ignore[list-item]

        try:
            # iterator over payloads with index
            it = iter(enumerate(payloads))

            # initial in-flight tasks
            in_flight: set[asyncio.Task] = set()
            for _ in range(min(parallelism, total)):
                i, payload = next(it)
                t = asyncio.create_task(run_one(i, payload))
                t._idx = i  # type: ignore[attr-defined]
                in_flight.add(t)

            while in_flight:
                done, in_flight = await asyncio.wait(
                    in_flight, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    idx = getattr(task, "_idx", None)
                    try:
                        idx, res = task.result()
                    except Exception as e:
                        if idx is None:
                            raise
                        unique_id = None
                        try:
                            unique_id = payloads[idx].get("unique_id")
                        except Exception:
                            unique_id = None
                        raise RuntimeError(
                            f"Remote verify failed for sample idx={idx}, unique_id={unique_id}: {e}"
                        ) from e
                    else:
                        results[idx] = res

                    progress.update(task_id, advance=1)

                    # replenish one task if payloads remain
                    try:
                        i, payload = next(it)
                    except StopIteration:
                        continue
                    else:
                        t_new = asyncio.create_task(run_one(i, payload))
                        t_new._idx = i  # type: ignore[attr-defined]
                        in_flight.add(t_new)

        finally:
            progress.stop()
            if executor is not None:
                executor.shutdown(wait=True)

        return results

    def _aggregate_group_scores(
        self, payloads: List[Dict[str, Any]], scores: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group rollouts by unique_id, normalize rm_score_raw within each group, and compute final score per sample.
        Returns a list aligned with `payloads` order containing enriched score dicts.
        """
        if len(payloads) != len(scores):
            raise ValueError(f"payload/score length mismatch: {len(payloads)} vs {len(scores)}")

        grouped: Dict[str, List[int]] = defaultdict(list)
        for idx, payload in enumerate(payloads):
            grouped[payload["unique_id"]].append(idx)

        updated = [None] * len(scores)  # type: ignore[list-item]

        for idxs in grouped.values():
            result = _aggregate_group_task((idxs, [scores[i] for i in idxs]))
            for global_idx, s in result:
                updated[global_idx] = s

        if any(item is None for item in updated):
            raise ValueError("Failed to aggregate all group scores; missing entries detected.")

        # type: ignore[return-value]
        return updated

    @staticmethod     
    def stable_sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def __del__(self) -> None:
        shutdown_fn = getattr(self, "_shutdown_async", None)
        if shutdown_fn is None:
            return
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(shutdown_fn())
        except Exception:
            pass
        finally:
            loop.close()
