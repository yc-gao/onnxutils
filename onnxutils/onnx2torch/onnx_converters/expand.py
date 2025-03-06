from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter


class TorchExpand(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data, shape):
        return data.expand(*shape)


@add_converter(op_type='Expand', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    torch_module = TorchExpand()
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
