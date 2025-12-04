"""
DeepSpeed + Zeus 多卡 GPT LoRA 训练脚本.

支持:
- DeepSpeed ZeRO-2/3 分布式训练
- Zeus GPU 能耗监控和优化
- PEFT LoRA 参数高效微调
- INT8/INT4 量化
"""

import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Zeus 可选导入
try:
    from zeus.monitor import ZeusMonitor
    from zeus.optimizer.power_limit import HFGlobalPowerLimitOptimizer

    ZEUS_AVAILABLE = True
except ImportError:
    ZEUS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 配置类
# ============================================================


@dataclass
class ModelConfig:
    """模型配置."""

    model_name: str = "EleutherAI/gpt-neo-1.3B"
    tokenizer_name: Optional[str] = None
    load_in_8bit: bool = True
    load_in_4bit: bool = False
    trust_remote_code: bool = True


@dataclass
class LoRAConf:
    """LoRA 配置."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    """数据配置."""

    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_length: int = 512
    num_workers: int = 4


@dataclass
class TrainConfig:
    """训练配置."""

    output_dir: str = "./outputs"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    bf16: bool = True
    gradient_checkpointing: bool = True
    deepspeed: Optional[str] = "configs/ds_config_zero2.json"
    use_zeus: bool = True
    seed: int = 42


# ============================================================
# 参数解析
# ============================================================


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="DeepSpeed + Zeus LLM Training")

    # 模型参数
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--load_in_8bit", action="store_true", default=True)
    parser.add_argument("--no_8bit", dest="load_in_8bit", action="store_false")
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # 数据参数
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--max_length", type=int, default=512)

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default="configs/ds_config_zero2.json")
    parser.add_argument("--seed", type=int, default=42)

    # Zeus 参数
    parser.add_argument("--use_zeus", action="store_true", default=True)
    parser.add_argument("--no_zeus", dest="use_zeus", action="store_false")

    args = parser.parse_args()

    # 处理空字符串
    if args.deepspeed == "":
        args.deepspeed = None

    model_config = ModelConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    lora_conf = LoRAConf(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    data_config = DataConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        max_length=args.max_length,
    )

    train_config = TrainConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        deepspeed=args.deepspeed,
        use_zeus=args.use_zeus,
        seed=args.seed,
    )

    return model_config, lora_conf, data_config, train_config


# ============================================================
# Zeus 回调
# ============================================================


def create_zeus_callback(train_config: TrainConfig):
    """创建 Zeus 回调."""
    if not train_config.use_zeus:
        return []

    if not ZEUS_AVAILABLE:
        logger.warning("Zeus 不可用，跳过能耗优化")
        return []

    try:
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            logger.warning("无 GPU，跳过 Zeus")
            return []

        gpu_indices = list(range(gpu_count))
        monitor = ZeusMonitor(gpu_indices=gpu_indices)
        optimizer = HFGlobalPowerLimitOptimizer(monitor)
        logger.info(f"Zeus 已启用，监控 GPU: {gpu_indices}")
        return [optimizer]
    except Exception as e:
        logger.warning(f"Zeus 初始化失败: {e}")
        return []


# ============================================================
# GPU 显存日志
# ============================================================


def log_gpu_memory():
    """记录 GPU 显存使用."""
    if not torch.cuda.is_available():
        return

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        logger.info(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


# ============================================================
# 模型加载
# ============================================================


def load_model_and_tokenizer(model_config: ModelConfig, lora_conf: LoRAConf):
    """加载模型和 tokenizer."""
    tokenizer_name = model_config.tokenizer_name or model_config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 量化配置
    quantization_config = None
    if model_config.load_in_8bit or model_config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=model_config.load_in_8bit,
            load_in_4bit=model_config.load_in_4bit,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=quantization_config,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16,
    )

    # LoRA
    peft_config = LoraConfig(
        r=lora_conf.r,
        lora_alpha=lora_conf.lora_alpha,
        lora_dropout=lora_conf.lora_dropout,
        target_modules=lora_conf.target_modules,
        bias=lora_conf.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================
# 数据准备
# ============================================================


def prepare_dataset(data_config: DataConfig, tokenizer):
    """准备数据集."""
    dataset = load_dataset(data_config.dataset_name, data_config.dataset_config)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=data_config.max_length,
            padding="max_length",
        )

    # 过滤空文本
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    tokenized = dataset.map(tokenize_fn, batched=True, num_proc=data_config.num_workers)

    return tokenized.get("train"), tokenized.get("validation")


# ============================================================
# 主函数
# ============================================================


def main():
    """主训练流程."""
    model_config, lora_conf, data_config, train_config = parse_args()
    set_seed(train_config.seed)

    logger.info("加载模型和 tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_config, lora_conf)
    log_gpu_memory()

    logger.info("准备数据集...")
    train_dataset, eval_dataset = prepare_dataset(data_config, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Zeus 回调
    callbacks = create_zeus_callback(train_config)

    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        num_train_epochs=train_config.num_train_epochs,
        max_steps=train_config.max_steps,
        warmup_ratio=train_config.warmup_ratio,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        eval_steps=train_config.eval_steps,
        bf16=train_config.bf16,
        gradient_checkpointing=train_config.gradient_checkpointing,
        deepspeed=train_config.deepspeed,
        seed=train_config.seed,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    logger.info("开始训练...")
    trainer.train()
    log_gpu_memory()

    logger.info("保存模型...")
    trainer.save_model()
    logger.info("训练完成")


if __name__ == "__main__":
    main()
