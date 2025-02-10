from typing import Tuple
from typing import NamedTuple

import torch


class OnnxToTorchModule:
    pass


class OnnxMapping:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class OperationConverterResult(NamedTuple):
    torch_module: torch.nn.Module
    onnx_mapping: OnnxMapping

