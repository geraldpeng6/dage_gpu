"""DeepSpeed + Zeus 训练脚本单元测试."""

import sys

from src.train import DataConfig
from src.train import LoRAConf
from src.train import ModelConfig
from src.train import TrainConfig
from src.train import create_zeus_callback
from src.train import log_gpu_memory
from src.train import parse_args


class TestModelConfig:
    """ModelConfig 测试."""

    def test_default_values(self):
        """默认值测试."""
        config = ModelConfig()
        assert config.model_name == "EleutherAI/gpt-neo-1.3B"
        assert config.load_in_8bit is True

    def test_custom_values(self):
        """自定义值测试."""
        config = ModelConfig(model_name="gpt2", load_in_8bit=False)
        assert config.model_name == "gpt2"


class TestLoRAConf:
    """LoRAConf 测试."""

    def test_default_values(self):
        """默认值测试."""
        config = LoRAConf()
        assert config.r == 16
        assert config.lora_alpha == 32

    def test_custom_rank(self):
        """自定义 rank 测试."""
        config = LoRAConf(r=8, lora_alpha=16)
        assert config.r == 8


class TestDataConfig:
    """DataConfig 测试."""

    def test_default_dataset(self):
        """默认数据集测试."""
        config = DataConfig()
        assert config.dataset_name == "wikitext"

    def test_boundary_max_length(self):
        """边界值测试."""
        config = DataConfig(max_length=1)
        assert config.max_length == 1


class TestTrainConfig:
    """TrainConfig 测试."""

    def test_default_values(self):
        """默认值测试."""
        config = TrainConfig()
        assert config.per_device_train_batch_size == 2
        assert config.use_zeus is True

    def test_disable_zeus(self):
        """禁用 Zeus 测试."""
        config = TrainConfig(use_zeus=False)
        assert config.use_zeus is False


class TestParseArgs:
    """parse_args 函数测试."""

    def test_default_args(self, monkeypatch):
        """默认参数测试."""
        monkeypatch.setattr(sys, "argv", ["train.py"])
        model_cfg, lora_cfg, data_cfg, train_cfg = parse_args()
        assert model_cfg.model_name == "EleutherAI/gpt-neo-1.3B"

    def test_custom_model(self, monkeypatch):
        """自定义模型测试."""
        monkeypatch.setattr(sys, "argv", ["train.py", "--model_name", "gpt2"])
        model_cfg, _, _, _ = parse_args()
        assert model_cfg.model_name == "gpt2"

    def test_disable_zeus_flag(self, monkeypatch):
        """禁用 Zeus 标志测试."""
        monkeypatch.setattr(sys, "argv", ["train.py", "--no_zeus"])
        _, _, _, train_cfg = parse_args()
        assert train_cfg.use_zeus is False

    def test_lora_params(self, monkeypatch):
        """LoRA 参数测试."""
        monkeypatch.setattr(
            sys, "argv", ["train.py", "--lora_r", "8", "--lora_alpha", "16"]
        )
        _, lora_cfg, _, _ = parse_args()
        assert lora_cfg.r == 8


class TestCreateZeusCallback:
    """Zeus 回调测试."""

    def test_zeus_disabled(self):
        """禁用 Zeus 返回空列表."""
        config = TrainConfig(use_zeus=False)
        callbacks = create_zeus_callback(config)
        assert callbacks == []


class TestLogGpuMemory:
    """GPU 显存日志测试."""

    def test_no_crash(self):
        """不崩溃测试."""
        log_gpu_memory()


class TestBoundaryValues:
    """边界值测试."""

    def test_zero_max_steps(self, monkeypatch):
        """max_steps=0 测试."""
        monkeypatch.setattr(sys, "argv", ["train.py", "--max_steps", "0"])
        _, _, _, train_cfg = parse_args()
        assert train_cfg.max_steps == 0

    def test_negative_batch_size(self, monkeypatch):
        """负数 batch_size 测试."""
        monkeypatch.setattr(
            sys, "argv", ["train.py", "--per_device_train_batch_size", "-1"]
        )
        _, _, _, train_cfg = parse_args()
        assert train_cfg.per_device_train_batch_size == -1


class TestFailureCases:
    """异常情况测试."""

    def test_invalid_type(self):
        """无效类型测试."""
        config = LoRAConf(r="invalid")  # type: ignore
        assert config.r == "invalid"

    def test_empty_model_name(self, monkeypatch):
        """空模型名测试."""
        monkeypatch.setattr(sys, "argv", ["train.py", "--model_name", ""])
        model_cfg, _, _, _ = parse_args()
        assert model_cfg.model_name == ""
