"""pytest fixtures."""

import sys
from unittest.mock import MagicMock

# Mock 外部依赖 (在测试前)
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.device_count.return_value = 0
mock_torch.cuda.memory_allocated.return_value = 0
mock_torch.cuda.memory_reserved.return_value = 0
mock_torch.bfloat16 = MagicMock()

mock_transformers = MagicMock()
mock_peft = MagicMock()
mock_peft.TaskType = MagicMock()
mock_peft.TaskType.CAUSAL_LM = "CAUSAL_LM"
mock_datasets = MagicMock()
mock_zeus = MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["transformers"] = mock_transformers
sys.modules["peft"] = mock_peft
sys.modules["datasets"] = mock_datasets
sys.modules["zeus"] = mock_zeus
sys.modules["zeus.monitor"] = MagicMock()
sys.modules["zeus.optimizer"] = MagicMock()
sys.modules["zeus.optimizer.power_limit"] = MagicMock()
