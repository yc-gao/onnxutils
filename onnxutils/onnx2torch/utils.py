from typing import Tuple
from typing import NamedTuple

import torch


class OnnxToTorchModule:
    pass


class OnnxMapping(NamedTuple):
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]


class OperationConverterResult(NamedTuple):
    torch_module: torch.nn.Module
    onnx_mapping: OnnxMapping
