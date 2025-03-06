import torch
from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter


class TorchReshape(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data, shape):
        return torch.reshape(data, shape)


@add_converter(op_type='Reshape', version=14)
def _(onnx_node: OnnxNode, _: OnnxModel):
    allowzero: bool = bool(onnx_node.attrs.get('allowzero', 0))

    assert not allowzero, 'not implement'

    torch_module = TorchReshape()
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
