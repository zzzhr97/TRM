#!/usr/bin/env python3
"""
Remote verifier client that reuses the original answer-extraction logic
and talks to an OpenAI-compatible sGLang server. Exposes a `compute_score`
function so Hydra's custom_reward_function can import it directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import atexit
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import httpx
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("remote_verifier.py requires httpx. Install it via `pip install httpx`.") from exc

from verl.utils.reward_score.custom_math_verify import extract_solution


LOGGER = logging.getLogger("remote_verifier")
VERIFIER_PASS_TAG = "Final Decision: Yes"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 360
DEFAULT_PARALLEL_LIMIT = 128
DEFAULT_VERIFIER_BACKENDS = "100.97.200.2:23888"

DEFAULT_RM_MODEL = "RewardModel"
DEFAULT_RM_TIMEOUT = 720
DEFAULT_RM_PARALLEL_LIMIT = 64
DEFAULT_RM_API_KEY = "None"
DEFAULT_RM_BACKENDS = "100.97.200.2:39999"
DEFAULT_RM_POSTFIX = "/v1"

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHAT_TEMPLATE_PATH = REPO_ROOT / "chat_templates" / "verifier.jinja"

def _raise_data_error(message: str, *, data: Any | None = None) -> None:
    full = f"{message} | data={data}" if data is not None else message
    LOGGER.warning(full)
    print(f"[remote_verifier][WARNING] {full}")
    raise ValueError(message)


def _log_warning(message: str) -> None:
    LOGGER.warning(message)
    print(f"[remote_verifier][WARNING] {message}")


def _ensure_no_proxy(hosts: tuple[str, ...]) -> None:
    """
    Append given hosts to NO_PROXY/no_proxy to avoid proxying verifier/RM calls.
    """
    if not hosts:
        return
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    entries = [h.strip() for h in existing.split(",") if h.strip()]
    for host in hosts:
        if host not in entries:
            entries.append(host)
    joined = ",".join(entries)
    os.environ["NO_PROXY"] = joined
    os.environ["no_proxy"] = joined


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Chat template not found: {path}") from None



def _resolve_setting(key: str, kwargs: Dict[str, Any], env_key: str, default: Any) -> Any:
    """
    Simplified resolver: only accepts explicit kwargs overrides; ignores environment variables.
    """
    if key in kwargs and kwargs[key] is not None:
        return kwargs[key]
    return default


def _parse_backends(raw: Optional[str], default_host: str, default_port: int) -> tuple[tuple[str, int], ...]:
    if not raw:
        return ((default_host, default_port),)
    items = []
    for entry in str(raw).split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 2:
            _log_warning(f"Invalid backend entry '{entry}', expected host:port.")
            continue
        host = parts[0]
        try:
            port = int(parts[1])
        except ValueError:
            _log_warning(f"Invalid backend port in '{entry}'.")
            continue
        items.append((host, port))
    if not items:
        return ((default_host, default_port),)
    return tuple(items)


@dataclass(frozen=True)
class VerifierEndpoint:
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/chat/completions"


@dataclass(frozen=True)
class VerifierSettings:
    host: str
    port: int
    endpoints: tuple[VerifierEndpoint, ...]
    model: str
    chat_template_path: Path
    temperature: float
    max_tokens: int
    timeout: int
    parallel_limit: int

    @property
    def base_url(self) -> str:
        return self.endpoints[0].base_url


@dataclass(frozen=True)
class RewardModelEndpoint:
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}{DEFAULT_RM_POSTFIX}"


@dataclass(frozen=True)
class RewardModelSettings:
    host: str
    port: int
    endpoints: tuple[RewardModelEndpoint, ...]
    model: str
    timeout: int
    parallel_limit: int
    api_key: str
    retry_limit: int
    retry_interval: float

    @property
    def base_url(self) -> str:
        return self.endpoints[0].base_url


def _build_settings(kwargs: Dict[str, Any]) -> VerifierSettings:
    host_default = "localhost"
    port_default = 25292
    raw_backends = kwargs.get("verifier_backends") or DEFAULT_VERIFIER_BACKENDS
    backends = _parse_backends(raw_backends, host_default, port_default)
    _ensure_no_proxy(tuple({h for h, _ in backends}))
    host = backends[0][0]
    port = backends[0][1]
    endpoints = tuple(VerifierEndpoint(host=h, port=p) for h, p in backends)
    model = _resolve_setting("model", kwargs, "REMOTE_VERIFIER_MODEL", "TIGER-Lab/general-verifier")
    template_path = Path(
        _resolve_setting("chat_template_path", kwargs, "REMOTE_VERIFIER_CHAT_TEMPLATE", str(DEFAULT_CHAT_TEMPLATE_PATH))
    ).expanduser()
    temperature = float(_resolve_setting("temperature", kwargs, "REMOTE_VERIFIER_TEMPERATURE", DEFAULT_TEMPERATURE))
    max_tokens = int(_resolve_setting("max_tokens", kwargs, "REMOTE_VERIFIER_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    timeout = int(_resolve_setting("timeout", kwargs, "REMOTE_VERIFIER_TIMEOUT", DEFAULT_TIMEOUT))
    parallel = int(_resolve_setting("parallel_limit", kwargs, "REMOTE_VERIFIER_PARALLEL", DEFAULT_PARALLEL_LIMIT))
    return VerifierSettings(
        host=host,
        port=port,
        endpoints=endpoints,
        model=model,
        chat_template_path=template_path,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        parallel_limit=max(1, parallel),
    )


def _build_reward_model_settings(
    kwargs: Dict[str, Any],
) -> Optional[RewardModelSettings]:
    model = _resolve_setting("reward_model_name", kwargs, "REMOTE_RM_MODEL", DEFAULT_RM_MODEL)
    if not model:
        return None
    host_default = "localhost"
    port_default = 25292
    raw_backends = kwargs.get("reward_model_backends") or DEFAULT_RM_BACKENDS
    backends = _parse_backends(raw_backends, host_default, port_default)
    _ensure_no_proxy(tuple({h for h, _ in backends}))
    host = backends[0][0]
    port = backends[0][1]
    timeout = int(_resolve_setting("reward_model_timeout", kwargs, "REMOTE_RM_TIMEOUT", DEFAULT_RM_TIMEOUT))
    parallel_limit = int(
        _resolve_setting(
            "reward_model_parallel_limit",
            kwargs,
            "REMOTE_RM_PARALLEL",
            DEFAULT_RM_PARALLEL_LIMIT,
        )
    )
    api_key = str(_resolve_setting("reward_model_api_key", kwargs, "REMOTE_RM_API_KEY", DEFAULT_RM_API_KEY))
    retry_limit = int(
        _resolve_setting("reward_model_retry_limit", kwargs, "REMOTE_RM_RETRY_LIMIT", 8)
    )
    retry_interval = float(
        _resolve_setting("reward_model_retry_interval", kwargs, "REMOTE_RM_RETRY_INTERVAL", 0.1)
    )
    return RewardModelSettings(
        host=str(host),
        port=port,
        endpoints=tuple(RewardModelEndpoint(host=h, port=p) for h, p in backends),
        model=str(model),
        timeout=timeout,
        parallel_limit=max(1, parallel_limit),
        api_key=api_key,
        retry_limit=max(1, retry_limit),
        retry_interval=max(0.0, retry_interval),
    )


def _extract_reasoning_trace(response: str) -> str:
    if not response:
        return ""
    lower = response
    if "<think>" in lower:
        try:
            lower = lower.split("<think>", 1)[1]
        except Exception:  # pragma: no cover
            return ""
    if "</think>" in lower:
        try:
            lower = lower.split("</think>", 1)[0]
        except Exception:  # pragma: no cover
            return ""
    return lower.strip()


def _extract_question(extra_info: Optional[Dict[str, Any]]) -> str:
    info = extra_info or {}
    llm_info = info.get("raw_llm_verifier") or {}
    code_info = info.get("raw_code_tests") or {}
    question = (
        llm_info.get("question")
        or code_info.get("question")
        or info.get("question")
    )
    if not question:
        _raise_data_error("[remote_verifier] Missing question in extra_info", data=extra_info)
    return str(question)

def _extract_prompt(extra_info: Optional[Dict[str, Any]]) -> str:
    if not extra_info:
        _raise_data_error("[remote_verifier] extra_info missing while extracting user_prompt")
    prompt = extra_info.get("user_prompt")
    if not prompt:
        _raise_data_error("[remote_verifier] extra_info missing 'user_prompt'", data=extra_info)
    return str(prompt)


class RemoteVerifierClient:
    def __init__(self, settings: VerifierSettings):
        self.settings = settings
        self.chat_template = _read_file(settings.chat_template_path)

    async def verify(
        self,
        question: str,
        ground_truth: str,
        student_answer: str,
    ) -> Tuple[bool, str]:
        if not student_answer:
            return False, "[remote_verifier] empty student answer"
        
        student_answer = student_answer[-2000:]

        messages = [
            {
                "role": "user",
                "content": (
                    "User: ### Question: {question}\n\n"
                    "### Ground Truth Answer: {ground_truth}\n\n"
                    "### Student Answer: {student_answer}\n\n"
                    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
                    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
                    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
                ).format(question=question, ground_truth=ground_truth, student_answer=student_answer),
            }
        ]
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "messages": messages,
            "extra_body": {
                "chat_template": self.chat_template,
            },
        }

        endpoint = _select_verifier_endpoint(self.settings)
        limiter = _get_verifier_semaphore(endpoint, self.settings)
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            async with limiter:
                try:
                    async with httpx.AsyncClient(timeout=self.settings.timeout) as client:
                        response = await client.post(endpoint.base_url, json=payload)
                    if response.status_code != 200:
                        snippet = response.text[:2000]
                        raise RuntimeError(f"HTTP {response.status_code}: {snippet}")
                    body = response.json()
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt >= max_attempts:
                        raise RuntimeError(
                            f"[remote_verifier] verifier request failed after {attempt} attempts "
                            f"({endpoint.host}:{endpoint.port}): {exc}"
                        ) from exc
                    await asyncio.sleep(0.2 * attempt)

        decision_blob = _extract_response_text(body)
        return VERIFIER_PASS_TAG in decision_blob, decision_blob


def _extract_response_text(body: Dict[str, Any]) -> str:
    choices = body.get("choices")
    if choices and isinstance(choices, list):
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        if content:
            return content
        if "text" in choice:
            return choice["text"]
    _log_warning(f"Unexpected response payload from verifier: {body}")
    raise ValueError(f"[remote_verifier] Unexpected response payload: {body}")


class RewardModelClient:
    def __init__(self, settings: RewardModelSettings):
        self.settings = settings
        kind = (os.getenv("REWARD_MODEL") or "trm").strip().lower()
        if kind not in {"trm", "reasonflux"}:
            _log_warning(f"[remote_verifier] Unknown REWARD_MODEL '{kind}', defaulting to trm")
            kind = "trm"
        self.reward_model_kind = kind

    def _format_input(
        self,
        prompt: str,
        ground_truth: str,
        solution_str: str,
    ) -> str:
        reasoning = _extract_reasoning_trace(solution_str or "") or (solution_str or "")
        return f"{prompt or ''}\n{reasoning}".strip()

    async def _embed_async_trm(self, text: str, endpoint: RewardModelEndpoint) -> tuple[float, str]:
        payload = {
            "model": self.settings.model,
            "input": text,
        }
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        async with httpx.AsyncClient(timeout=self.settings.timeout) as client:
            response = await client.post(f"{endpoint.base_url}/embeddings", headers=headers, json=payload)
            if response.status_code != 200:
                snippet = response.text[:2000]
                _log_warning(f"Reward model error {response.status_code}: {snippet}")
                raise ValueError(f"[remote_verifier] reward model HTTP {response.status_code}")
            body = response.json()
        data = body.get("data")
        if not data:
            _log_warning("Reward model returned empty embedding data.")
            raise ValueError("[remote_verifier] Reward model returned empty embedding data.")
        embedding = data[0].get("embedding")
        if not embedding:
            _log_warning("Reward model embedding missing 'embedding' field.")
            raise ValueError("[remote_verifier] Reward model embedding missing data.")
        value = embedding[0]
        return float(value), ""

    def _format_reasonflux_messages(self, prompt: str, response: str) -> list[list[dict]]:
        completion = response.replace("\n\n", "<extra_0>") + "<extra_0>"
        return [
            [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": completion}]},
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": response + '<extra_0>'}]},
            ],
        ]

    async def _score_reasonflux_messages(self, messages: list[dict], endpoint: RewardModelEndpoint) -> float:
        payload = {"model": "ReasonFlux-PRM", "messages": messages}
        headers = {"Authorization": f"Bearer {self.settings.api_key}"} if self.settings.api_key else {}
        url = f"{endpoint.base_url}/pooling"
        async with httpx.AsyncClient(timeout=self.settings.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                snippet = response.text[:2000]
                _log_warning(f"ReasonFlux reward model error {response.status_code}: {snippet}")
                raise ValueError(f"[remote_verifier] reward model HTTP {response.status_code}")
            body = response.json()
        if body.get("object") != "list":
            raise ValueError(f"ReasonFlux PRM response object mismatch: {body.get('object')}")
        data = body.get("data") or []
        if not data or data[0].get("object") != "pooling":
            raise ValueError("ReasonFlux PRM response missing pooling data")
        pool_data = data[0].get("data")
        if not isinstance(pool_data, list) or not pool_data:
            raise ValueError("ReasonFlux PRM pooling data malformed")
        # pool_data should be [[p0, p1], ...]
        probs: list[float] = []
        for row in pool_data:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError(f"ReasonFlux PRM pooling unexpected row: {row}")
            probs.append(float(row[1]))
        if not probs:
            raise ValueError("ReasonFlux PRM pooling produced no probabilities")
        return sum(probs) / len(probs)

    async def _embed_async_reasonflux(
        self,
        prompt: str,
        response: str,
        endpoint: RewardModelEndpoint,
    ) -> tuple[float, str]:
        message_variants = self._format_reasonflux_messages(prompt, response)
        total = 0.0
        for messages in message_variants:
            total += await self._score_reasonflux_messages(messages, endpoint)
        total /= 2.0 # (0.0, 2.0) -> (0.0, 1.0)
        return total, ""

    async def _embed_async(
        self,
        text: str,
        endpoint: RewardModelEndpoint,
        *,
        prompt: str = "",
        response: str = "",
    ) -> tuple[float, str]:
        if self.reward_model_kind == "reasonflux":
            return await self._embed_async_reasonflux(prompt, response, endpoint)
        return await self._embed_async_trm(text, endpoint)

    async def score(
        self,
        prompt: str,
        ground_truth: str,
        solution_str: str,
    ) -> tuple[float, str]:
        text = self._format_input(prompt, ground_truth, solution_str or "")
        endpoint = _select_rm_endpoint(self.settings)
        limiter = _get_rm_semaphore(endpoint, self.settings)
        max_attempts = max(1, min(3, self.settings.retry_limit))
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            async with limiter:
                try:
                    score, blob = await self._embed_async(
                        text,
                        endpoint,
                        prompt=prompt,
                        response=solution_str or "",
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt >= max_attempts:
                        raise RuntimeError(
                            f"[remote_verifier] reward model failed after {attempt} attempts "
                            f"({endpoint.host}:{endpoint.port}): {exc}"
                        ) from exc
                    await asyncio.sleep(self.settings.retry_interval or 0.1)

        # from uuid import uuid4
        # with open(f"/root/projects/ReasoningFaithfulness/main_rl/main/outputs/rm_input/{str(uuid4())}.txt", "w", encoding="utf-8") as f:
        #     f.write(text + "\n\n" + str(score))

        return score, blob


_CLIENT_CACHE: Dict[VerifierSettings, RemoteVerifierClient] = {}
_RM_CLIENT_CACHE: Dict[RewardModelSettings, RewardModelClient] = {}
_VERIFIER_MULTI_LOCK = threading.Lock()
_VERIFIER_MULTI_COUNTER = 0
_RM_MULTI_LOCK = threading.Lock()
_RM_MULTI_COUNTER = 0
_VERIFIER_SEMAPHORES: Dict[tuple[int, str, int, int], asyncio.Semaphore] = {}
_RM_SEMAPHORES: Dict[tuple[int, str, int, int], asyncio.Semaphore] = {}


def _get_client(settings: VerifierSettings) -> RemoteVerifierClient:
    client = _CLIENT_CACHE.get(settings)
    if client is None:
        client = RemoteVerifierClient(settings)
        _CLIENT_CACHE[settings] = client
    return client


def _get_rm_client(settings: RewardModelSettings) -> RewardModelClient:
    client = _RM_CLIENT_CACHE.get(settings)
    if client is None:
        client = RewardModelClient(settings)
        _RM_CLIENT_CACHE[settings] = client
    return client


def _verifier_semaphore_key(endpoint: VerifierEndpoint, settings: VerifierSettings) -> tuple[int, str, int, int]:
    loop = asyncio.get_running_loop()
    return (id(loop), endpoint.host, endpoint.port, settings.parallel_limit)


def _get_verifier_semaphore(endpoint: VerifierEndpoint, settings: VerifierSettings) -> asyncio.Semaphore:
    key = _verifier_semaphore_key(endpoint, settings)
    semaphore = _VERIFIER_SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(settings.parallel_limit)
        _VERIFIER_SEMAPHORES[key] = semaphore
    return semaphore


def _rm_semaphore_key(endpoint: RewardModelEndpoint, settings: RewardModelSettings) -> tuple[int, str, int, int]:
    loop = asyncio.get_running_loop()
    return (id(loop), endpoint.host, endpoint.port, settings.parallel_limit)


def _get_rm_semaphore(endpoint: RewardModelEndpoint, settings: RewardModelSettings) -> asyncio.Semaphore:
    key = _rm_semaphore_key(endpoint, settings)
    semaphore = _RM_SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(settings.parallel_limit)
        _RM_SEMAPHORES[key] = semaphore
    return semaphore


def _select_verifier_endpoint(settings: VerifierSettings) -> VerifierEndpoint:
    endpoints = settings.endpoints
    if len(endpoints) == 1:
        return endpoints[0]
    global _VERIFIER_MULTI_COUNTER
    with _VERIFIER_MULTI_LOCK:
        idx = _VERIFIER_MULTI_COUNTER % len(endpoints)
        _VERIFIER_MULTI_COUNTER += 1
    return endpoints[idx]


def _select_rm_endpoint(settings: RewardModelSettings) -> RewardModelEndpoint:
    endpoints = settings.endpoints
    if len(endpoints) == 1:
        return endpoints[0]
    global _RM_MULTI_COUNTER
    with _RM_MULTI_LOCK:
        idx = _RM_MULTI_COUNTER % len(endpoints)
        _RM_MULTI_COUNTER += 1
    return endpoints[idx]


async def _async_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    kwargs = dict(kwargs)

    raw_rm = None
    rm_blob = None
    rm_settings = _build_reward_model_settings(kwargs)
    if rm_settings is None:
        _log_warning("Reward model settings are missing; cannot compute RM score.")
    else:
        raw_rm, rm_blob = await _async_reward_model_score(
            rm_settings=rm_settings,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

    verify_blob = None
    verdict, decision_blob = await _async_llm_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    verify_score = 1.0 if verdict else 0.0
    verify_blob = decision_blob
    verify_missing_student_answer = decision_blob == "[remote_verifier] missing student answer"

    if raw_rm is None:
        raise ValueError("[remote_verifier] reward model score missing.")

    result = {
        # "score": composition["final_score"],
        "verify_score": verify_score,
        "verify_binary": 1.0 if verify_score >= 1.0 else 0.0,
        # "rm_score": composition.get("rm_score"),
        "rm_score_raw": raw_rm,
        "verify_missing_student_answer": verify_missing_student_answer,
    }
    if verify_blob is not None:
        result["verify_blob"] = verify_blob
    if rm_blob is not None:
        result["rm_blob"] = rm_blob
    return result


async def _async_llm_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> tuple[bool, str]:
    extra_info = extra_info or {}
    if extra_info.get("math_verify_score") == 1:
        return True, "[remote_verifier] math_verify_pass"

    # only apply llm verifier for webinstruct data
    if not data_source in ["TIGER-Lab/WebInstruct-verified"]:
        # olympiads, cn_contest, aops_forum, amc_aime, inequalities, olympiads_ref, number_theory
        return False, "[remote_verifier] only applies math-verify to non-webinstruct data"

    question = _extract_question(extra_info)
    student_answer, extract_err = extract_solution(solution_str or "")

    if student_answer is None:
        return False, "[remote_verifier] missing student answer"

    # length penalty for long answers
    len_answer = len(student_answer)
    len_gt = len(ground_truth)
    if len_answer - len_gt >= max(0.5 * len_gt, 10):
        return False, "[remote_verifier] length penalty"

    settings = _build_settings(kwargs)
    client = _get_client(settings)
    try:
        verdict, decision_blob = await client.verify(
            question, ground_truth, student_answer
        )
    except Exception as exc:
        raise RuntimeError(
            f"[remote_verifier] verify failed for {data_source} ({settings.host}:{settings.port}): {exc}"
        ) from exc

    return verdict, decision_blob


async def _async_reward_model_score(
    rm_settings: RewardModelSettings,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[float], Optional[str]]:
    prompt = _extract_prompt(extra_info)
    client = _get_rm_client(rm_settings)
    try:
        score, blob = await client.score(
            prompt=prompt,
            ground_truth=ground_truth,
            solution_str=solution_str or "",
        )
    except Exception as exc:
        raise RuntimeError(
            f"[remote_verifier] reward model failed ({rm_settings.host}:{rm_settings.port}): {exc}"
        ) from exc
    if score is None:
        raise ValueError("[remote_verifier] Reward model returned None score.")
    return score, blob or None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synchronous wrapper so Hydrated configs can call it directly.
    """
    try:
        return asyncio.run(
            _async_compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **kwargs,
            )
        )
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                _async_compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **kwargs,
                )
            )
        finally:
            loop.close()


async def _shutdown_clients() -> None:
    _CLIENT_CACHE.clear()
    _RM_CLIENT_CACHE.clear()


# expose async implementation so reward managers can detect and batch calls
compute_score.async_fn = _async_compute_score  # type: ignore[attr-defined]
compute_score.shutdown_async_fn = _shutdown_clients  # type: ignore[attr-defined]
compute_score._verify_rm_source = "remote_verifier"  # type: ignore[attr-defined]


def shutdown() -> None:
    """
    Optional helper to close aiohttp sessions when running this module standalone.
    """
    try:
        asyncio.run(_shutdown_clients())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_shutdown_clients())
        finally:
            loop.close()


def _atexit_shutdown() -> None:
    try:
        shutdown()
    except Exception:  # pragma: no cover - best-effort cleanup
        LOGGER.debug("remote_verifier shutdown at exit failed", exc_info=True)


atexit.register(_atexit_shutdown)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual check for remote verifier connectivity.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--student-answer", required=True)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--chat-template-path", default=None)
    args = parser.parse_args()

    score = compute_score(
        data_source="manual",
        solution_str=args.student_answer,
        ground_truth=args.ground_truth,
        extra_info={"question": args.question},
        host=args.host,
        port=args.port,
        model=args.model,
        chat_template_path=args.chat_template_path,
    )
    print(f"Verifier score: {score}")
