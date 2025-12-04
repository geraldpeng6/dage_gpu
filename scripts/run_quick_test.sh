#!/bin/bash
# 快速测试脚本 - 验证环境和显存
# 使用方式: bash scripts/run_quick_test.sh

set -e

echo "=== 检查 GPU ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB')
"

echo ""
echo "=== 检查依赖 ==="
python -c "
import transformers
import peft
import deepspeed
import accelerate
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
print(f'deepspeed: {deepspeed.__version__}')
print(f'accelerate: {accelerate.__version__}')

try:
    from zeus.monitor import ZeusMonitor
    print('zeus: available')
except ImportError:
    print('zeus: NOT available (pip install zeus-ml)')
"

echo ""
echo "=== 快速训练测试 (50 步, gpt2-small) ==="
python src/train.py \
    --model_name "gpt2" \
    --max_steps 50 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_length 128 \
    --output_dir "./outputs/test" \
    --deepspeed "" \
    --no_zeus

echo ""
echo "=== 测试完成! ==="
