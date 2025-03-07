from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter


class TorchMatmul(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x0, x1):
        return x0 @ x1


@add_converter(op_type='MatMul', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    torch_module = TorchMatmul()
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
