#!/usr/bin/env bash
set -euo pipefail

echo "Running as $(whoami)"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Environment settings
# ---------------------------------------------------------------------------
export TRANSFORMERS_OFFLINE=1
export WANDB_DIR=<local-path>
export CUDA_VISIBLE_DEVICES="0,1,2,3"

N_GPUS_PER_NODE=4
TRAIN_BATCH_SIZE=512
LEARNING_RATE=5e-7
PPO_MINI_BATCH_SIZE=128
CLIP_RATIO_HIGH=0.30
CLIP_RATIO_LOW=0.30
CLIP_RATIO_C=10
KL_LOSS_COEF=0.0001
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE=low_var_kl
TEMPERATURE=1.0
KL_COEF=0.000

RUN_TAG=0210
DATASET_NAME=zzzhr97--WebInstruct-Verified-Processed
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192

MODEL_TAG=L8I
HDFS_MODEL_PATH=<local-path>
HDFS_DATA_PATH=<local-path>
HDFS_CHECKPOINT_PATH=<local-path>
HDFS_LOG_PATH=<local-path>

RUN_NAME="${RUN_TAG}_${MODEL_TAG}_bs${N_GPUS_PER_NODE}x${TRAIN_BATCH_SIZE}-${PPO_MINI_BATCH_SIZE}"

LOG_FILE_PATH="${HDFS_LOG_PATH}/${RUN_NAME}.log"
mkdir -p "$(dirname "${LOG_FILE_PATH}")"

cat <<EOF
------------------------------
Run name:          ${RUN_NAME}
Prompt/Response:   ${MAX_PROMPT_LENGTH} / ${MAX_RESPONSE_LENGTH}
------------------------------
EOF

CMD=(
  python -u main_ppo.py
  "algorithm.adv_estimator=grpo"
  "custom_reward_function.path=$(pwd)/remote_verifier.py"
  "custom_reward_function.name=compute_score"
  "reward_model.reward_manager=verify"
  "reward_model.enable=False"
  "reward_model.launch_reward_fn_async=True"
  "+reward_model.reward_kwargs.parallelism=1024"
  "data.return_raw_chat=True"
  "data.train_files=[${HDFS_DATA_PATH}/${DATASET_NAME}/train.parquet]"
  "data.val_files=[${HDFS_DATA_PATH}/${DATASET_NAME}/test.parquet]"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.val_batch_size=500"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "actor_rollout_ref.model.path=${HDFS_MODEL_PATH}"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
  "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFFIENT}"
  "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}"
  "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}"
  "actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}"
  "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=2"
  "actor_rollout_ref.rollout.max_num_batched_tokens=65536"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.70"
  "actor_rollout_ref.rollout.n=8"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536"
  "algorithm.kl_ctrl.kl_coef=${KL_COEF}"
  "trainer.critic_warmup=0"
  "trainer.logger=['console','wandb','tensorboard']"
  "trainer.project_name=TRM-Reasoner"
  "trainer.experiment_name=${RUN_NAME}"
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.nnodes=1"
  "trainer.save_freq=5"
  "trainer.test_freq=10"
  "trainer.default_local_dir=${HDFS_CHECKPOINT_PATH}/${RUN_NAME}"
  "trainer.total_epochs=2"
  "trainer.val_before_train=True"
)

echo "Launching training..."
HYDRA_FULL_ERROR=1 "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE_PATH}"
