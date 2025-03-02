from torch import nn
from onnxutils.onnx import OnnxNode, OnnxModel

from ..converter_registry import add_converter


class TorchIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


@add_converter(op_type='Identity', version=16)
def _(onnx_node: OnnxNode, _: OnnxModel):  # pylint: disable=unused-argument
    torch_module = TorchIdentity()
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
