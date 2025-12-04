# DeepSpeed + Zeus Multi-GPU Training

基于 **DeepSpeed** (分布式训练) + **Zeus** (能耗优化) 的 4×4080 GPT LoRA 训练项目。

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
uv venv --python 3.10
source .venv/bin/activate

# 安装依赖
uv sync

# 或使用 pip
pip install -e .
```

### 2. 验证安装

```bash
python -c "
import torch
import deepspeed
from zeus.monitor import ZeusMonitor
print(f'PyTorch: {torch.__version__}')
print(f'DeepSpeed: {deepspeed.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print('Zeus imported successfully')
"
```

### 3. 运行训练

```bash
# 单机 4 卡训练 (DeepSpeed ZeRO-2 + Zeus)
accelerate launch --config_file configs/accelerate_config.yaml \
    src/train.py \
    --model_name EleutherAI/gpt-neo-2.7B \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8
```

## 项目结构

```text
dage_gpu/
├── docs/
│   └── PLAN.md              # 详细计划文档
├── configs/
│   ├── ds_config_zero2.json # DeepSpeed ZeRO-2 配置
│   └── accelerate_config.yaml
├── src/
│   └── train.py             # 训练脚本
├── pyproject.toml           # 依赖配置
└── README.md
```

## 显存配置

| 模型 | ZeRO Stage | 量化 | 单卡显存 | 状态 |
|------|------------|------|----------|------|
| GPT-Neo-1.3B | ZeRO-2 | INT8 | ~4 GB | ✅ 安全 |
| GPT-Neo-2.7B | ZeRO-2 | INT8 | ~8 GB | ✅ 推荐 |
| Llama-2-7B | ZeRO-3 | INT8 | ~12 GB | ⚠️ 紧张 |

## 文档

- [详细计划](docs/PLAN.md) - 显存分析、Zeus 集成、实施步骤

## 参考

- [Zeus GitHub](https://github.com/ml-energy/zeus)
- [DeepSpeed](https://www.deepspeed.ai/)
- [PEFT/LoRA](https://huggingface.co/docs/peft)
