from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchMatMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0, x1):
        return x0 @ x1


@add_converter(op_type='MatMul', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    torch_module = TorchMatMul()
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
