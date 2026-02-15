# Copyright 2025 Individual Contributor: Thibaut Barroyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import multiprocessing
import os
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Optional

import ray
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import get_reward_manager_cls
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        config (dict): Configuration dictionary containing custom_reward_function
                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    function_name = reward_fn_config.get("name")
    assert function_name is not None

    path_obj = Path(file_path).expanduser().resolve()
    module_dir = str(path_obj.parent)
    module_name = reward_fn_config.get("module_name") or path_obj.stem

    # ensure module dir is importable for child processes (Ray workers)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    existing_py_path = os.environ.get("PYTHONPATH", "")
    py_paths = [p for p in existing_py_path.split(os.pathsep) if p] if existing_py_path else []
    if module_dir not in py_paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([module_dir, *py_paths])

    module = sys.modules.get(module_name)
    if module is None:
        if not path_obj.exists():
            raise FileNotFoundError(f"Reward function file '{path_obj}' not found.")

        spec = importlib.util.spec_from_file_location(module_name, str(path_obj))
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[module_name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except Exception as e:
            # clean up partially registered module to avoid stale state
            sys.modules.pop(module_name, None)
            raise RuntimeError(f"Error loading module from '{path_obj}': {e}") from e
    else:
        existing_path = getattr(module, "__file__", None)
        if existing_path and Path(existing_path).resolve() != path_obj:
            raise RuntimeError(
                f"Module name '{module_name}' already loaded from '{existing_path}', "
                f"cannot reuse for '{path_obj}'. Please set a unique module_name."
            )

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{module.__file__}'.")

    print(f"using customized reward function '{function_name}' from module '{module_name}' -> '{module.__file__}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    wrapped = partial(_call_with_kwargs, raw_fn, reward_kwargs)
    # ===== preserve compute_score attributes for VerifyRewardManager =====
    for attr in ("_verify_rm_source", "async_fn", "shutdown_async_fn"):
        if hasattr(raw_fn, attr):
            setattr(wrapped, attr, getattr(raw_fn, attr))
    # ===== preserve compute_score attributes for VerifyRewardManager =====

    async_fn = getattr(raw_fn, "async_fn", None)
    if async_fn is not None:

        async def wrapped_async_fn(*args, **kwargs):
            merged = {**kwargs, **reward_kwargs}
            return await async_fn(*args, **merged)

        wrapped.async_fn = wrapped_async_fn  # type: ignore[attr-defined]

    shutdown_async_fn = getattr(raw_fn, "shutdown_async_fn", None)
    if shutdown_async_fn is not None:
        wrapped.shutdown_async_fn = shutdown_async_fn  # type: ignore[attr-defined]

    return wrapped


def load_reward_manager(
    config: DictConfig, tokenizer: Any, num_examine: int, **reward_kwargs: Any
) -> AbstractRewardManager:
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """

    # Try to get a custom reward function based on the configuration
    # user defined reward manager can be registered in custom_reward_fn
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024)
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn: AbstractRewardManager) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    if reward_fn is None:
        assert config is not None and tokenizer is not None, (
            "config and tokenizer must not be None when reward_fn is None"
        )

        warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )

    return compute_reward(data, reward_fn)

#### Reward worker thread for threading-based reward computation ####
import threading
import queue
import itertools
import traceback

class RewardThread:
    """
    单线程后台工作者：在独立线程中计算 reward，避免 Ray actor/multiprocessing。
    """

    def __init__(self, *, config=None, tokenizer=None, reward_fn=None):
        self.in_q = queue.Queue(maxsize=128)
        self.out_q = queue.Queue(maxsize=128)
        self._stop_event = threading.Event()
        self._req_id_gen = itertools.count(1)

        # 保留初始化参数，线程内构建 reward_fn（或复用传入的 reward_fn）
        self._init_config = config
        self._init_tokenizer = tokenizer
        self._reward_fn = reward_fn

        self.thread = threading.Thread(
            target=self._run,
            name="RewardThread",
            daemon=False,  # 允许线程内再调度/新建子任务
        )
        self.thread.start()

    def _ensure_reward_fn(self):
        if self._reward_fn is not None:
            return self._reward_fn
        config = self._init_config
        tokenizer = self._init_tokenizer
        if config is None or tokenizer is None:
            raise RuntimeError("RewardThread missing config/tokenizer to build reward_fn.")
        self._reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        return self._reward_fn

    def _run(self):
        while not self._stop_event.is_set():
            try:
                item = self.in_q.get()
                if item is None:
                    break

                req_id, data = item
                try:
                    reward_fn = self._ensure_reward_fn()
                    result = compute_reward(data, reward_fn)
                    self.out_q.put((req_id, True, result))
                except Exception:
                    self.out_q.put((req_id, False, traceback.format_exc()))
            except Exception:
                # 防止线程 silently die
                traceback.print_exc()

    def submit(self, data) -> int:
        req_id = next(self._req_id_gen)
        self.in_q.put((req_id, data))
        return req_id

    def get(self, req_id: int, timeout: float | None = None):
        while True:
            rid, ok, payload = self.out_q.get(timeout=timeout)
            if rid != req_id:
                # 单线程场景正常不会乱序，保险起见直接放回队首
                self.out_q.put((rid, ok, payload))
                continue
            if ok:
                return payload
            raise RuntimeError(f"Reward worker thread error: {payload}")

    def close(self):
        try:
            self._stop_event.set()
            self.in_q.put(None)
        finally:
            self.thread.join(timeout=2)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
#### Reward worker thread for threading-based reward computation ####
