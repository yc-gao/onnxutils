import torch
from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter


class TorchCumSum(nn.Module):
    def __init__(self, axis) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return torch.cumsum(x, self.axis)


@add_converter(op_type='CumSum', version=14)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    exclusive: int = onnx_node.attrs.get('exclusive', 0)
    reverse: int = onnx_node.attrs.get('reverse', 0)

    assert exclusive == 0, 'not implement'
    assert reverse == 0, 'not implement'

    axis = onnx_model.get_initializer_by_name(
        onnx_node.input_names[1]).item()

    torch_module = TorchCumSum(axis)
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
