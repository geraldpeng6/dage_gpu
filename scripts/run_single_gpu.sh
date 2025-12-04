#!/bin/bash
# 单卡 4090 测试脚本

set -e

# 4090 24GB 可以用更大的 batch_size
BATCH_SIZE=${BATCH_SIZE:-4}
MODEL=${MODEL:-"EleutherAI/gpt-neo-1.3B"}

echo "=== 单卡 4090 训练测试 ==="
echo "模型: $MODEL"
echo "Batch Size: $BATCH_SIZE"

# 检查 GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"

# 运行训练 (不用 DeepSpeed，单卡不需要)
accelerate launch --config_file configs/accelerate_single_gpu.yaml \
    src/train.py \
    --model_name "$MODEL" \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 4 \
    --max_steps 50 \
    --deepspeed "" \
    --output_dir ./outputs/single_gpu_test

echo "=== 测试完成 ==="
