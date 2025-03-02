import torch
from torch import nn
from onnxutils.onnx import OnnxNode, OnnxModel

from ..converter_registry import add_converter

func_mapping = {
    'Abs': torch.abs,
    'Neg': torch.neg,
    'Sqrt': torch.sqrt,
    'Exp': torch.exp,
    'Floor': torch.floor,
    'Tanh': torch.tanh,
    'Atan': torch.atan,
    'Cos': torch.cos,
    'Sin': torch.sin,
}


class TorchUnaryOp(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


@add_converter(op_type='Abs', version=13)
@add_converter(op_type='Neg', version=13)
@add_converter(op_type='Sqrt', version=13)
@add_converter(op_type='Exp', version=13)
@add_converter(op_type='Floor', version=13)
@add_converter(op_type='Tanh', version=13)
@add_converter(op_type='Atan', version=7)
@add_converter(op_type='Cos', version=7)
@add_converter(op_type='Sin', version=7)
def _(onnx_node: OnnxNode, _: OnnxModel):  # pylint: disable=unused-argument
    torch_module = TorchUnaryOp(func_mapping[onnx_node.op_type])
    onnx_mapping = {
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
