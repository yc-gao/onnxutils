import torch
from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter


class TorchSplit(nn.Module):
    def __init__(self, axis) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, data, split):
        return torch.split(data, split, dim=self.axis)


@add_converter(op_type='Split', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    axis: int = onnx_node.attrs.get('axis', 0)

    torch_module = TorchSplit(axis)
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
