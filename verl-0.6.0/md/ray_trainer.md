# Ray PPO Trainer 批处理流水线详解

> 结合 `verl/trainer/ppo/ray_trainer.py`（主训练循环）以及 `verl/protocol.py`、`verl/trainer/ppo/core_algos.py`、`verl/trainer/ppo/reward.py` 等关联模块，梳理单个 `batch_dict` 在 Ray 单控制器 PPO 流水线中的完整旅程。

## 1. 数据注入与 DataProto 载体
1. `StatefulDataLoader` 产出的 `batch_dict` 被 `DataProto.from_single_dict` 包装（`ray_trainer.py:1018-1025`），自动将张量字段移入 `batch`，将字符串/JSON 等放入 `non_tensor_batch`，并初始化 `meta_info`。`DataProto` 会在构造时调用 `check_consistency` 确保 batch 维度一致（`protocol.py:328-486`）。
2. 训练循环给每条样本分配 `uid`（UUID 字符串）写入 `non_tensor_batch`，后续 advantage 计算、日志和 replay importance sampling 都依赖此唯一键。
3. `_get_gen_batch`（`ray_trainer.py:502-517`）从原始批次中 `pop` 生成阶段暂不需要的 `input_ids`/`attention_mask` 等字段，仅保留提示和奖励需要的非张量信息，生成一个更轻的 `gen_batch`。
4. `gen_batch` 的 `meta_info` 写入当前 `global_steps`，并用 `repeat(interleave=True)` 扩展到 `rollout.n` 份提示，确保一个 batch 可以生成多条响应（`ray_trainer.py:1027-1035`）。

## 2. 生成阶段（Rollout）
1. `actor_rollout_wg.generate_sequences`（或异步 `async_rollout_manager`）接收 `gen_batch`，在 Ray worker 上运行采样推理，返回包含 `responses`、`logits`、`rollout_log_probs`、`token_level_scores` 等字段的 `DataProto`（`ray_trainer.py:1037-1055`）。
2. 如果 `algorithm.adv_estimator` 配置为 REMAX，则会复制一个 `gen_baseline_batch`，设置 `do_sample=False` 生成确定性响应，再通过 `reward_fn` 估计 baseline 奖励并存入 `batch.batch["reward_baselines"]`（`ray_trainer.py:1045-1062`）。
3. 主 batch 经 `repeat` 与生成结果 `union` 后，若缺失 `response_mask` 则调用 `compute_response_mask` 基于 `attention_mask` 和 `responses` 补齐（`ray_trainer.py:1064-1070`）。
4. 若开启 `trainer.balance_batch`，`_balance_batch` 会按全局 token 数对样本重排，使每个 DP rank 分配到的 token 更均衡（`ray_trainer.py:1071-1086, 900-915`）。

## 3. 奖励与外部信息融合
1. 奖励模型路径：若 `reward_model.enable`，`rm_wg.compute_rm_score` 先写入 `rm_scores`，在缺少 SFT 先验时可直接复用奖励模型输出（`ray_trainer.py:1091-1095`）。
2. Rule-based / 自定义奖励：根据配置调用 `compute_reward` 或 `compute_reward_async`（`ray_trainer.py:1096-1110`，`reward.py:118-175`），得到 `reward_tensor`（token 级分数）与 `reward_extra_infos_dict`（如 `reward_model.ground_truth`、sandbox 返回的诊断信息）；后者会被转换成 numpy 数组塞回 `non_tensor_batch`，随 batch 发往更新 worker。
3. KL 惩罚：若 `algorithm.use_kl_in_reward`，则需要先准备 `ref_log_prob`（见下一节），再由 `apply_kl_penalty` 用当前策略与参考策略的 `kl` 调整 `token_level_rewards` 并更新自适应 KL 系数（`ray_trainer.py:129-167, 1135-1141`）。否则 `token_level_scores` 会直接复制到 `token_level_rewards`。

## 4. 统计复算（Log prob / 参考策略 / 价值）
1. **Old log probs**：调用 `actor_rollout_wg.compute_log_prob` 重算 `old_log_probs`、`entropys`，并用 `agg_loss` 按 `loss_agg_mode` 聚合成 `actor/entropy` 指标。返回的 DataProto 中临时字段会被 `pop` 后 `union` 回 batch（`ray_trainer.py:1100-1113`）。如果 rollout 时也记录了 `rollout_log_probs`，此处还会调用 `calculate_debug_metrics` 对比 rollout 与当前策略。
2. **Reference policy**：当配置 `need_reference_policy(config)` 且 `use_kl_in_reward=True` 时，`ref_policy_wg` 或 actor 自身会计算 `ref_log_prob`，写回 batch 供 KL 罚和 mismatch 统计使用（`ray_trainer.py:1114-1123`）。
3. **Critic values**：若 `need_critic(config)`，则 `critic_wg.compute_values` 产生 `values`（通常对应 state value 或 token value），直接 `union` 回 batch，供 advantage 估计（`ray_trainer.py:1124-1131`）。

## 5. Rollout Importance Sampling（可选）
- 当 `algorithm.rollout_is_threshold` 不为空且 batch 中含 `rollout_log_probs` 时，`compute_rollout_importance_weights_and_add_to_batch` 会调用 `compute_rollout_importance_weights`（`mismatch_helper.py`）算出 IS 权重及 mismatch 指标（KL/PPL/weight 统计），并根据 `algorithm.rollout_is` 决定是否把 `rollout_is_weights` `union` 到 batch（`ray_trainer.py:922-955, 1152-1158`）。这样可以在 rollout 策略与训练策略存在偏差时校正梯度，或仅记录监控信息。

## 6. 优势与目标值计算
1. `compute_advantage` 是 driver 端的轻量操作：若 batch 缺 `response_mask` 会补齐，然后根据 `algorithm.adv_estimator` 选择 GAE、GRPO、REINFORCE++、RLOO、OPO 等实现（`ray_trainer.py:196-259` + `core_algos.py:88-220` 及后续函数）。
2. 该函数会写入 `advantages` 与 `returns`；GRPO/RLOO 等估计器会使用 `non_tensor_batch["uid"]` 对重复样本做聚合，REMAX 会结合 `reward_baselines`，而 GRPO 还会根据 `norm_adv_by_std_in_grpo` 决定是否做标准差归一化。
3. 如果配置了 `rollout_is_weights`，它们也会在 advantage 计算时被透传，供后续 actor loss 读取。

## 7. Critic / Actor 更新与日志
1. **Critic 更新**：`critic_wg.update_critic(batch)` 在各自的 Ray worker 内执行多步优化，并把 loss、梯度范数、学习率等写入其返回值的 `meta_info["metrics"]`，driver 端通过 `reduce_metrics` 聚合（`ray_trainer.py:1160-1168`）。
2. **Actor 更新**：在 `trainer.critic_warmup` 之后才会调用 `actor_rollout_wg.update_actor(batch)`；调用前会把 `meta_info["multi_turn"]`（源自 `actor_rollout_ref.rollout.multi_turn`）写入 batch，供 worker 调整前向逻辑（`ray_trainer.py:1169-1176`）。
3. **Rollout 数据落盘**：若 `trainer.rollout_data_dir` 配置非空，`_log_rollout_data` 会把 prompts、responses、token-level scores、ground truth、request_id 等拼成 JSONL 存盘，帮助后评与问题定位（`ray_trainer.py:447-466, 1190-1198`）。
4. **验证与 checkpoint**：根据 `trainer.test_freq` 调 `_validate()`（内部会复用 pad→generate→reward→metric 的流程），并依据 `trainer.save_freq` 或 ESI 过期时间触发 `_save_checkpoint`。这些动作都在同一 `timing_raw` 计时器中打点，方便 `compute_timing_metrics` 做阶段统计（`ray_trainer.py:1183-1216`）。
5. **指标汇总**：每次 step 会聚合 `compute_data_metrics`（token、长度、reward）与 `compute_throughout_metrics`（tflop/s、samples/s）以及 mismatch/entropy/KL 等，交给 `Tracking` logger 上报；如 sampler 支持 curriculum，还会调用 `sampler.update(batch)` 根据表现调节数据分布（`ray_trainer.py:1254-1270`）。

## 8. 关键辅助组件一览
- **DataProto 工具箱**：`repeat`/`chunk`/`union` 等方法（`protocol.py:871-1107`）支撑 batch 在 driver 与 worker 间的分发、重排与序列化。序列化可选 `torch.save` 或 numpy buffer，配合 Ray 对象存储传输。
- **Reward Pipeline**：`reward.py` 中的 `load_reward_manager` 支持自定义脚本、sandbox fusion、Prime/DAPO reward manager 等多种实现，可通过配置指定 `reward_manager`、`custom_reward_function.path/name` 等。
- **Advantage Registry**：`core_algos.py` 用 `register_adv_est` 把各类 advantage 函数挂到 `ADV_ESTIMATOR_REGISTRY`，外部也可注册新名字；KL 控制器（`AdaptiveKLController`/`FixedKLController`）则由 `apply_kl_penalty` 间接使用。
- **Mismatch 工具**：`compute_rollout_importance_weights` 除了给出 IS 权重，还会返回 `mismatch/kl`、`mismatch/ppl`、`mismatch/weight_*` 等指标，便于监控 rollout 策略质量。

---
通过上述分层拆解，可以看到单个 batch 在 Ray PPO Trainer 中经历了 “DataProto 化 → 多路 rollout → 奖励/参考策略对齐 → IS/KL 调整 → Advantage 聚合 → Critic/Actor 更新 → 指标 & 资产落盘” 的完整闭环，每个步骤都以 `DataProto` 为总线，保证张量与附加元信息在 driver 与各 WorkerGroup 之间同步、可扩展、可追踪。
