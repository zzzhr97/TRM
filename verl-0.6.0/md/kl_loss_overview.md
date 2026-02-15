# actor\_rollout\_ref.actor.use\_kl\_loss 机制说明

## 配置入口
- `actor_rollout_ref.actor.use_kl_loss` 定义在 `verl/trainer/config/_generated_ppo_trainer.yaml`（布尔值，默认 `false`）。GRPO/LLaMA-RL 的脚本通常将其显式设置为 `true`。
- 当该开关为 `true` 时，`need_reference_policy(config)` 会要求加载参考策略，并在数据 batch 中注入 `ref_log_prob`（`verl/trainer/ppo/ray_trainer.py:1114-1135`）。

## 数据流
1. **参考策略对数概率**
   - `compute_ref_log_prob` 分别在 Megatron/FSDP rollout worker 中实现（`verl/workers/megatron_workers.py:712-732`, `verl/workers/fsdp_workers.py:992-1013`），输出 token 级别的 `ref_log_prob`。
   - `ray_trainer` 在计算旧 log-prob 之后调用上述接口，把结果 `union` 回 batch。
2. **Minibatch 选择**
   - `verl/workers/actor/megatron_actor.py:303-333` 在 `make_minibatch_iterator` 中，如果 `config.use_kl_loss=True`，就把 `ref_log_prob` 加入 `select_keys`，确保后续每个 mini-batch 都可访问参考 log-prob。

## 损失函数
- 在 `forward_backward_batch` 的 `loss_func` 内（`verl/workers/actor/megatron_actor.py:450-489`），总损失写作：

  ```math
  L = L_{\text{PG}} - \beta \, \mathbb{E}[H] + \lambda \, \mathrm{KL}(\pi_{\theta}\,\|\,\pi_{\text{ref}})
  ```

  其中：
  - `L_pg` 是 PPO/GRPO 的 policy gradient（含 clip ratio 等）；
  - `\beta` 为 `entropy_coeff`，系数来自 `meta_info`；
  - `\lambda = actor\_rollout\_ref.actor.kl\_loss\_coef`；
  - KL 项通过 `kl_penalty()` 计算，支持 `k1/k2/k3/low_var_kl/abs` 等多种估计（见 `verl/trainer/ppo/core_algos.py:1369-1420`），随后用 `agg_loss()` 按 `loss_agg_mode` 聚合成标量。
  - 如果配置 `kl_loss_type` 带 `+`（如 `k3+`），实现会采用 straight-through trick 以保证梯度无偏。

- 代码片段（`verl/workers/actor/megatron_actor.py:482-489`）：

  ```python
  if self.config.use_kl_loss:
      ref_log_prob = data["ref_log_prob"]
      kld = kl_penalty(logprob=log_prob,
                       ref_logprob=ref_log_prob,
                       kl_penalty=self.config.kl_loss_type)
      kl_loss = agg_loss(kld, response_mask, self.config.loss_agg_mode)
      policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
  ```

## 与 `algorithm.use_kl_in_reward` 的关系
- `actor_rollout_ref.actor.use_kl_loss` 控制的是 **actor loss** 里的 KL 正则。
- 如果同时启用 `algorithm.use_kl_in_reward`，`ray_trainer` 还会在 reward 侧调用 `apply_kl_penalty()`，把 KL 差分并入 token-level rewards（`verl/trainer/ppo/ray_trainer.py:396-412, 1135-1141`）。此时需要确保参考策略仅计算一次即可满足两个用途。
- 官方文档建议：GRPO 设 `use_kl_loss=True`、`algorithm.use_kl_in_reward=False`；DrGRPO 则关闭 KL loss，仅保留 reward 归一化。

## Clip Ratio（策略截断）
- `actor_rollout_ref.actor.clip_ratio` 对应 PPO 中的剪切阈值 `\varepsilon`，在 `compute_policy_loss()`（`verl/trainer/ppo/core_algos.py:838-888`）中体现。基本损失：

  ```math
  \mathcal{L}_{\text{clip}} = \mathbb{E}\big[\min\big(r_t A_t,\; \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) A_t\big)\big],
  ```

  其中 `r_t = \exp(\log \pi_\theta - \log \pi_{\text{old}})`。

- `actor_rollout_ref.actor.clip_ratio_c` 实现 Dual-Clip PPO（同文件第 869-877 行）。当 `A_t < 0` 时，再按

  ```math
  L = \min\big(\mathcal{L}_{\text{clip}}, -A_t \cdot c\big), \quad c = \text{clip\_ratio\_c} > 1,
  ```

  保证极端负优势情况下也有限制。代码在 `pg_losses3 = -advantages * clip_ratio_c` 以及后面的逐元素 `torch.where` 中实现；`pg_clipfrac_lower` 统计该分支命中率。

- 参数注入位置：`verl/workers/actor/megatron_actor.py:576-579` 把这两个值写入 `meta_info`，供 loss 计算使用。

## 小结
1. 开启 `use_kl_loss` 会强制准备参考策略并向 batch 注入 `ref_log_prob`。
2. Actor loss 变成 `PG − entropy + λ·KL`，其中 KL 的近似形式由 `kl_loss_type` 控制。
3. Clip Ratio 由 `clip_ratio`（标准 PPO 截断）与 `clip_ratio_c`（Dual-Clip 负优势下限）共同决定梯度范围。
4. 若还想在 reward 端做 KL 惩罚，需配合 `algorithm.use_kl_in_reward`，两者互不冲突，但会增加一次 KL 计算成本。 
