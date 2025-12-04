# DeepSpeed + Zeus 多卡 GPT LoRA 训练计划

## 项目背景

使用 **DeepSpeed** (分布式训练) + **Zeus** (能耗优化) 进行 4×4080 (64GB 总显存) 下的 GPT LoRA 训练。

### 框架分工

| 框架 | 职责 | 来源 |
|------|------|------|
| **DeepSpeed** | ZeRO 显存优化、分布式训练 | Microsoft |
| **Zeus** | GPU 能耗测量、Power Limit 自动优化 | [ml-energy/zeus](https://github.com/ml-energy/zeus) |
| **PEFT** | LoRA 参数高效微调 | HuggingFace |
| **Accelerate** | 训练启动器、DeepSpeed 集成 | HuggingFace |

### Zeus 核心组件

```python
from zeus.monitor import ZeusMonitor
from zeus.optimizer.power_limit import HFGlobalPowerLimitOptimizer

# 1. ZeusMonitor - 测量 GPU 能耗
monitor = ZeusMonitor(gpu_indices=[0, 1, 2, 3])

# 2. HFGlobalPowerLimitOptimizer - HuggingFace Trainer Callback
#    自动 profile 不同 power limit，选择最优配置
optimizer = HFGlobalPowerLimitOptimizer(monitor)

# 3. 集成到 Trainer (兼容 DeepSpeed)
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[optimizer],  # Zeus callback
)
```

---

## 硬件配置

| 项目 | 规格 |
|------|------|
| GPU | 4 × RTX 4080 |
| 单卡显存 | 16 GB |
| 总显存 | 64 GB |
| 拓扑 | 单机 4 卡 |

---

## 显存预算分析

### GPT3 模型规格 (Merak 支持的预定义配置)

| 模型 | 参数量 | FP16 显存 | INT8 显存 | INT4 显存 |
|------|--------|-----------|-----------|----------|
| gpt3-small | 125M | 0.25 GB | 0.13 GB | 0.06 GB |
| gpt3-medium | 350M | 0.7 GB | 0.35 GB | 0.18 GB |
| gpt3-large | 760M | 1.5 GB | 0.76 GB | 0.38 GB |
| gpt3-xl | 1.3B | 2.6 GB | 1.3 GB | 0.65 GB |
| gpt3-2.7b | 2.7B | 5.4 GB | 2.7 GB | 1.35 GB |
| gpt3-6.7b | 6.7B | 13.4 GB | 6.7 GB | 3.4 GB |
| gpt3-13b | 13B | 26 GB | 13 GB | 6.5 GB |
| gpt3-175b | 175B | 350 GB | 175 GB | 87.5 GB |

### 显存占用估算 (训练时)

**训练显存 ≈ 模型参数 + 梯度 + 优化器状态 + 激活值**

| 配置 | FP16 训练 (无优化) | + Activation Checkpointing | + LoRA (r=16) | + INT8 量化 |
|------|---------------------|---------------------------|---------------|-------------|
| gpt3-2.7b | ~22 GB | ~12 GB | ~8 GB | ~5 GB |
| gpt3-6.7b | ~54 GB | ~28 GB | ~16 GB | ~10 GB |
| gpt3-13b | ~104 GB | ~52 GB | ~30 GB | ~18 GB |

### 推荐配置 (4×16GB = 64GB 安全范围)

| 方案 | 模型 | ZeRO Stage | 量化 | 预估单卡显存 | 安全余量 |
|------|------|------------|------|-------------|----------|
| **方案 A (保守)** | GPT-Neo-1.3B | ZeRO-2 | INT8 | ~4 GB | ✅ 高 |
| **方案 B (推荐)** | GPT-Neo-2.7B | ZeRO-2 | INT8 | ~8 GB | ✅ 中 |
| **方案 C (激进)** | Llama-2-7B | ZeRO-3 + Offload | INT8 | ~12 GB | ⚠️ 紧张 |

---

## 技术架构

### DeepSpeed ZeRO 显存优化

```text
┌─────────────────────────────────────────────────────────┐
│              DeepSpeed ZeRO Optimization                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ZeRO-1: 优化器状态分片 (Optimizer State Partitioning)  │
│  ZeRO-2: + 梯度分片 (Gradient Partitioning)             │
│  ZeRO-3: + 参数分片 (Parameter Partitioning)            │
│                                                         │
│  4×4080 推荐: ZeRO-2 (平衡显存节省与通信开销)           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Zeus 能耗优化工作流

```text
┌─────────────────────────────────────────────────────────┐
│                 Zeus Power Optimization                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Profiling Phase: 测试不同 Power Limit (100W~320W)   │
│  2. Measurement: 记录每个配置的 时间 & 能耗             │
│  3. Selection: 选择最优 Power Limit                     │
│     - Energy: 最低能耗                                  │
│     - Time: 最短时间                                    │
│     - ZeusCost: 时间-能耗平衡 (η 参数可调)              │
│  4. Apply: 自动设置所有 GPU 的 Power Limit              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 4 卡数据并行方案

```text
DeepSpeed ZeRO-2 + 4 GPU Data Parallel

┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
│ GPU 0 │   │ GPU 1 │   │ GPU 2 │   │ GPU 3 │
│ Shard │   │ Shard │   │ Shard │   │ Shard │
│  1/4  │   │  2/4  │   │  3/4  │   │  4/4  │
└───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘
    │           │           │           │
    └───────────┴─────┬─────┴───────────┘
                      │
              All-Reduce 梯度同步
```

---

## DeepSpeed + Zeus + LoRA 集成

### 完整训练代码示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from zeus.monitor import ZeusMonitor
from zeus.optimizer.power_limit import HFGlobalPowerLimitOptimizer

# 1. 加载模型 (INT8 量化)
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-2.7B",
    load_in_8bit=True,
    device_map="auto",
)

# 2. 应用 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 约 0.1% 可训练参数

# 3. Zeus 能耗优化 (Trainer Callback)
monitor = ZeusMonitor(gpu_indices=[0, 1, 2, 3])
zeus_optimizer = HFGlobalPowerLimitOptimizer(monitor)

# 4. 训练参数 (DeepSpeed 通过 Accelerate 集成)
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    gradient_checkpointing=True,
    deepspeed="configs/ds_config_zero2.json",  # DeepSpeed 配置
)

# 5. 创建 Trainer (Zeus callback 自动优化 Power Limit)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[zeus_optimizer],  # Zeus 能耗优化
)

trainer.train()
```

---

## 显存安全控制

### 1. 量化策略

| 方法 | 显存节省 | 精度损失 | 推荐场景 |
|------|----------|----------|----------|
| FP16 | 50% | 无 | 默认 |
| INT8 (LLM.int8) | 75% | 极低 | **推荐** |
| INT4 (QLoRA) | 87.5% | 低 | 超大模型 |

### 2. 激活检查点 (Gradient Checkpointing)

```python
training_args = TrainingArguments(
    gradient_checkpointing=True,  # 启用，节省 ~30% 显存
)
```

### 3. 梯度累积

```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,      # 小 batch
    gradient_accumulation_steps=16,     # 累积到有效 batch=16×4卡=64
)
```

### 4. 显存监控

```python
import torch

def log_gpu_memory():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

---

## 依赖版本

```toml
[project]
dependencies = [
    "torch>=2.1.0",            # DeepSpeed 兼容
    "transformers>=4.36.0",    # 最新 Trainer API
    "datasets>=2.16.0",
    "accelerate>=0.25.0",      # DeepSpeed 集成
    "deepspeed>=0.12.0",       # ZeRO 优化
    "peft>=0.7.0",             # LoRA 支持
    "bitsandbytes>=0.41.0",    # INT8/INT4 量化
    "zeus-ml",                 # Zeus 能耗优化
]
```

---

## 实施步骤

### Phase 1: 环境搭建

- [ ] 1.1 创建 Python 3.10+ 虚拟环境
- [ ] 1.2 安装 PyTorch 2.x + CUDA
- [ ] 1.3 安装 DeepSpeed, Accelerate
- [ ] 1.4 安装 Zeus (`pip install zeus-ml`)
- [ ] 1.5 验证 GPU 可见性和 NCCL

### Phase 2: 模型准备

- [ ] 2.1 选择模型 (推荐 GPT-Neo-2.7B 或 Llama-2-7B)
- [ ] 2.2 配置 INT8 量化 (`load_in_8bit=True`)
- [ ] 2.3 配置 LoRA 适配器 (r=16, alpha=32)
- [ ] 2.4 验证显存占用

### Phase 3: 数据准备

- [ ] 3.1 下载数据集 (WikiText-2 / Alpaca)
- [ ] 3.2 预处理和分词
- [ ] 3.3 配置 DataLoader

### Phase 4: 训练配置

- [ ] 4.1 配置 DeepSpeed ZeRO-2
- [ ] 4.2 配置 gradient checkpointing
- [ ] 4.3 配置梯度累积
- [ ] 4.4 添加 Zeus 能耗监控

### Phase 5: 训练与验证

- [ ] 5.1 运行快速测试 (100 步)
- [ ] 5.2 监控显存和能耗
- [ ] 5.3 调整参数防止 OOM
- [ ] 5.4 完整训练

---

## 风险与应对

| 风险 | 可能性 | 影响 | 应对措施 |
|------|--------|------|----------|
| 显存 OOM | 中 | 高 | 逐步增加 batch，实时监控，启用 ZeRO-3 |
| NCCL 通信问题 | 低 | 中 | 检查 GPU 拓扑，使用 `NCCL_DEBUG=INFO` |
| Zeus 权限问题 (Power Limit) | 中 | 低 | 需要 root 或设置 `nvidia-smi -pm 1` |
| INT8 量化精度下降 | 低 | 低 | 监控 loss，必要时切换 FP16 |

---

## Zeus 权限设置

Zeus 需要修改 GPU Power Limit，需要以下权限：

```bash
# 方法 1: 使用 sudo 运行训练
sudo python train.py

# 方法 2: 设置 persistence mode (推荐)
sudo nvidia-smi -pm 1

# 方法 3: 使用 zeusd 守护进程 (无需 root)
# 参考: https://ml.energy/zeus/getting_started/#zeusd
```

---

## 参考资源

- [Zeus GitHub](https://github.com/ml-energy/zeus)
- [Zeus 文档](https://ml.energy/zeus/)
- [Zeus HuggingFace 集成示例](https://github.com/ml-energy/zeus/tree/master/examples/huggingface)
- [DeepSpeed 文档](https://www.deepspeed.ai/)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [bitsandbytes 量化](https://github.com/TimDettmers/bitsandbytes)
