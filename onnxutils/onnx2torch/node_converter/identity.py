from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from ..converter_registry import add_converter


class TorchIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


@add_converter(op_type='Identity', version=16)
def _(onnx_node: OnnxNode, _: OnnxModel):  # pylint: disable=unused-argument
    torch_module = TorchIdentity(),
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
