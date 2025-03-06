import torch
from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter


class TorchClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, min_f=None, max_f=None):
        return torch.clip(x, min_f, max_f)


@add_converter(op_type='Clip', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    torch_module = TorchClip()
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
