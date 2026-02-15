# Dynamic Batch Size (Dynamic BSZ) 机制详解

> 目标：在 GPU 显存固定、样本序列长度高度不均时，动态地按“**token 配额**”拆分 micro-batch，使每次前/反向都尽量“吃满”显存而不过量，从而提升吞吐、降低 OOM 风险。本节串联配置入口、Ray Trainer 注入逻辑、核心拆分算法以及各个 worker（Actor/Critic/Reward/Ref）的执行细节。

## 1. 开关与配置项
- **全局开关**：`actor_rollout_ref.actor.use_dynamic_bsz`（默认 `false`，`verl/trainer/config/actor/actor.yaml:22-33`）。开启后：
  - 训练阶段使用 `ppo_max_token_len_per_gpu`（默认 16384）作为每 GPU 的 token 上限。
  - 推理/old_log_prob 复算使用 `ppo_infer_max_token_len_per_gpu`。
- **其它角色复用**：
  - `critic.use_dynamic_bsz`、`reward_model.use_dynamic_bsz` 默认继承 actor 的开关，并引入 `ppo_max_token_len_per_gpu` / `forward_max_token_len_per_gpu`（`verl/trainer/config/critic/critic.yaml:52-85`、`verl/trainer/config/reward_model/reward_model.yaml:41-80`）。
  - `actor_rollout_ref.rollout.log_prob_use_dynamic_bsz` 与 `actor_rollout_ref.ref.log_prob_use_dynamic_bsz` 控制 rollout 模型和参考模型在 log-prob 计算阶段是否动态拆分（`verl/trainer/config/rollout/rollout.yaml:87-95`、`verl/trainer/config/ref/ref.yaml:16-20`）。
- **配置校验**：`validate_config` 仅在 *关闭* 动态 BSZ 时强制 `real_train_batch_size % minimal_bsz == 0`，并要求用户显式给出 `*_micro_batch_size_per_gpu`（`verl/utils/config.py:38-118`）。开启后可省去这些整除约束，依赖 token 配额自动切分。
- **示例脚本**：大模型 recipe（如 `examples/grpo_trainer/run_qwen3_vl-30b-megatron.sh:44-62`）会显式把 actor/ref/rollout 的动态开关都设为 `True`，并给出更大的 `ppo_max_token_len_per_gpu` 以覆盖多模态上下文。

## 2. Trainer 端：meta_info 注入 & DP 取样
- `ray_trainer.py:1018-1105` 中，DataLoader 产出的 `batch_dict` 被转成 `DataProto` 并赋予 `uid`；随后 `_get_gen_batch` 剪裁出生成所需字段。
- 在把 batch 分发给各 worker 前，Ray Trainer 会：
  - 计算 `global_token_num = sum(attention_mask)`（`ray_trainer.py:1078-1090`）供 MFU 统计。
  - 若配置了 `trainer.balance_batch`，调用 `_balance_batch` 先按 `attention_mask` 长度在 DP 内重排，保证各 rank 获得相似 token 量（`ray_trainer.py:900-915`）。
- 真正的动态拆分由 worker 完成，但 Trainer 会通过 RPC 调用时在 `meta_info` 中写入：
  - `use_dynamic_bsz`、`max_token_len_per_gpu` 或 `micro_batch_size_per_gpu`（`verl/workers/roles/actor.py:106-149`）。
  - `micro_batch_size`（Megatron DP 版本在 `verl/workers/megatron_workers.py:715-746`内设定）。

## 3. 核心算法：`rearrange_micro_batches`
- 实现位于 `verl/utils/seqlen_balancing.py:251-323`：
  1. 读取 `attention_mask`（或 nested `input_ids`），得到每样本有效 token 数 `seq_len_effective` 与长序列上界 `max_seq_len`；要求 `max_token_len >= max_seq_len`。
  2. 估算所需的 micro-batch 数 `ceil(total_tokens / max_token_len)`，并可以强制：
     - **DP 同步**：若 `same_micro_num_in_dp=True` 且传入 `dp_group`，使用 `all_reduce(max)` 保证所有 rank 按同样的 micro-batch 数切分，避免后续 `all_gather` deadlock。
     - **Pipeline 对齐**：若启用 Megatron 虚拟 PP，`num_batches_divided_by` 会把数量向上取整为 PP stage 的倍数，以满足流水线 scheduling。
     - **最小切片数**：`min_num_micro_batch` 在多阶段 forward/backward 中用于兜底。
  3. 使用 `get_seqlen_balanced_partitions` 把样本索引分配到 `num_micro_batches` 个 partition。`use_dynamic_bsz_balance` 开启后，会按“**序列长度平方和**”排序，以更精准地匹配注意力算子开销（`verl/utils/seqlen_balancing.py:304-314`）。
  4. 通过 `tensordict_utils.index_select_tensor_dict` 切出每个 micro-batch，并返回 `[TensorDict], [index_list]`。
- `prepare_dynamic_batch`（`verl/utils/seqlen_balancing.py:343-379`）包装上述结果，把每个 TensorDict + 非张量字段重新封装成 `DataProto`，供 worker 直接迭代。
- `restore_dynamic_batch`（`verl/utils/seqlen_balancing.py:382-404`）利用 `get_reverse_idx` 恢复原顺序，用于把 logprob/entropy/values 等输出拼回完整 batch。
- 单元/多机测试覆盖：`tests/utils/test_seqlen_balancing.py:14-120` 验证了单卡重建、`same_micro_num_in_dp`、`min_num_micro_batch` 等场景；`test_dynamic_batch` 也确保 `restore_dynamic_batch` 的正确性。

## 4. 各角色如何使用动态 BSZ
### 4.1 Actor（FSDP & Megatron）
- **Ref / rollout log-prob 推理**：
  - FSDP 路径在 `verl/workers/roles/actor.py:106-137` 设置 `max_token_len_per_gpu = ppo_infer_max_token_len_per_gpu`，并在 engine 推理前把 DataProto 转成 `TensorDict`。
  - Megatron DP 实现（`verl/workers/actor/dp_actor.py:318-356`）读取 `data.meta_info` 中的 `max_token_len`，调用 `prepare_dynamic_batch` 完成拆分，然后在 `log_probs`/`entropy` 拼回前用 `restore_dynamic_batch` 保序。
  - Megatron Hybrid Engine 则由 `verl/workers/megatron_workers.py:733-757` 负责，把 `log_prob_use_dynamic_bsz` 对应的 `max_token_len` 保存在 `meta_info`，后续复用 DP actor 的实现。
- **PPO 更新**：
  - Worker 先把一个全局 mini-batch 切作 `ppo_mini_batch_size`，再根据 `use_dynamic_bsz` 选择 `prepare_dynamic_batch` 或固定 `ppo_micro_batch_size_per_gpu`（`verl/workers/actor/dp_actor.py:365-420`）。
  - 为避免变长 micro-batch 改变梯度幅度，引入 `loss_scale_factor = (#response_tokens_in_micro_batch / ppo_mini_batch_size)`（`verl/workers/actor/dp_actor.py:417-421`），确保不同 micro-batch 对梯度贡献一致。
  - Ray 控制器会把 `global_token_num` 写入 `meta_info`，actor `update_actor` 用它估算 MFU（`verl/workers/roles/actor.py:150-185`）。

### 4.2 Critic
- 与 Actor 基本一致：`compute_values` 和 `update_critic` 分别在 `verl/workers/critic/dp_critic.py:152-244` 中使用 `prepare_dynamic_batch`，并同样通过 `loss_scale_factor` 归一化损失（`verl/workers/critic/dp_critic.py:237-244`）。默认的 token 配额更大（`ppo_max_token_len_per_gpu` 默认 32768），以覆盖值网络使用的 prompt+response 双端输入。

### 4.3 Reward Model
- FSDP/Ray worker 在 `verl/workers/fsdp_workers.py:1845-1886` 开启 `use_dynamic_bsz` 后会把 `forward_max_token_len_per_gpu * ulysses_sequence_parallel_size` 作为 `max_token_len`。
- Megatron RM（`verl/workers/reward_model/megatron/reward_model.py:120-262`）还需考虑虚拟 pipeline：`rearrange_micro_batches(..., num_batches_divided_by=microbatch_group_size_per_vp_stage)` 保证微批数量与 VP stage 匹配；输出 logits 会按 `indices` 逆序恢复。

### 4.4 参考策略 / rollout 引擎
- FSDP worker 在 `verl/workers/fsdp_workers.py:966-1008` 中设置 `data.meta_info["use_dynamic_bsz"] = config.rollout.log_prob_use_dynamic_bsz / config.ref.log_prob_use_dynamic_bsz`，其余逻辑与 actor 相同。
- Megatron worker 同理，参见 `verl/workers/megatron_workers.py:711-757`。

### 4.5 其它消费者
- 任何自定义 worker 只要读取 `data.meta_info` 中的 `use_dynamic_bsz`、`max_token_len`，调用 `prepare_dynamic_batch` / `restore_dynamic_batch` 即可复用该机制。

## 5. 输出恢复与指标缩放
- **顺序恢复**：每个使用动态 BSZ 的 compute 函数，都在输出阶段通过 `restore_dynamic_batch` 保证张量顺序与原 batch 完全一致，从而不影响基于 `uid` 的 advantage 聚合（例如 `verl/workers/actor/dp_actor.py:351-354`、`verl/workers/critic/dp_critic.py:182-188`）。
- **损失缩放**：Actor/Critic 的 `loss_scale_factor` 以“当前 micro-batch 行数 / ppo_mini_batch_size”度量，Reward Model 仅需在拼回 logits 前逆序；因此梯度在不同 micro-batch 间仍然等价于固定 batch-size 训练。

## 6. 使用流程总结
1. 在配置中将 `actor_rollout_ref.actor.use_dynamic_bsz=True`，并合理设置 `ppo_max_token_len_per_gpu`（经验值：`rollout.n * max_prompt_len + max_response_len`）。若 reference、rollout log-prob 和 reward/critic 也需动态拆分，保留默认的 `oc.select` 即可自动继承。
2. 验证阶段：`python scripts/generate_trainer_config.sh` 会持久化 `_generated_*.yaml`，方便确认各模块的 `*_use_dynamic_bsz` 与 token 配额。
3. 训练过程中注意：
   - `trainer.balance_batch=True` 可进一步平衡 DP rank 间的 token 分布，与动态微批互补。
   - 若在 Megatron 使用虚拟 PP，需要保证 `ppo_max_token_len_per_gpu` 与 `microbatch_group_size_per_vp_stage` 配合（`rearrange_micro_batches(..., num_batches_divided_by=...)`）。
4. 调试：查看 `metrics` 中的 `global_seqlen/*`（`_balance_batch` 会记录，`ray_trainer.py:900-915`）以及 `mismatch/` 指标，确认动态 BSZ 未导致某些 rank 长期吃不到长序列。

通过这些设计，动态 BSZ 让 verl 能够在面对超长上下文、多模态拼接、或批间长度差异极大的 RL 数据时，自动把 micro-batch 拆分成“token 等量”切片，既维持高 GPU 利用率，也避免了手工调 `micro_batch_size_per_gpu` 的反复试错。
