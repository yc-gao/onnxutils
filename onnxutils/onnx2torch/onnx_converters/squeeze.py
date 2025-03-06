import torch
from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter


class TorchSqueeze(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data, axes):
        return torch.squeeze(data, dim=axes)


@add_converter(op_type='Squeeze', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    torch_module = TorchSqueeze()
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
