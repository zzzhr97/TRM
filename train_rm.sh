DATASET_BASE_DIR=<local-path>
MODEL_PATH=<local-path>

num_gpus=8
per_device_train_bsz=2
per_device_test_bsz=16
gradient_accumulation_steps=12
num_epochs=1
learning_rate=5e-6
weight_decay=1e-3
warmup_ratio=0.03
eval_steps=8
save_steps=8
logging_steps=1
OUTPUT_DIR=<local-dir>

accelerate launch \
    --config_file=accelerate_configs/fsdp2.yaml \
    --num_processes $num_gpus \
    train.py \
    --model_name_or_path $MODEL_PATH \
    --train_file $DATASET_BASE_DIR/TRM-preference-train.json \
    --validation_file $DATASET_BASE_DIR/TRM-preference-test.json \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $per_device_train_bsz \
    --per_device_eval_batch_size $per_device_test_bsz \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --center_rewards_coefficient 0.001 \
    --num_train_epochs $num_epochs \
    --gradient_checkpointing True \
    --warmup_ratio $warmup_ratio \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --logging_strategy steps \
    --logging_steps $logging_steps \
    --logging_first_step True \
    --report_to wandb \
    --do_train True \
    --do_eval True \
    --eval_strategy steps \
    --eval_steps $eval_steps \
    --save_strategy steps \
    --save_steps $save_steps \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --seed 42 \
    --save_total_limit 1 \
    --dataloader_drop_last True \
    --greater_is_better True
