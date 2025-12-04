#!/bin/bash
# DeepSpeed + Zeus 训练启动脚本
# 使用方式: bash scripts/run_train.sh

set -e

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

# 训练参数
MODEL_NAME="${MODEL_NAME:-EleutherAI/gpt-neo-1.3B}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/gpt-neo-1.3B-lora}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-512}"

echo "=== DeepSpeed + Zeus Training ==="
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE x $GRAD_ACCUM (grad accum)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# 启动训练
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    src/train.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_train_epochs "$EPOCHS" \
    --max_length "$MAX_LENGTH"

echo "=== Training completed! ==="
