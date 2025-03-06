from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter


class TorchGemm(nn.Module):
    def __init__(self, alpha, beta, transA, transB) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB

    def forward(self, A, B, C):
        if self.transA:
            A = A.T
        if self.transB:
            B = B.T
        return A @ B * self.alpha + C * self.beta


@add_converter(op_type='Gemm', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    alpha = onnx_node.attrs.get('alpha', 1)
    beta = onnx_node.attrs.get('beta', 1)
    transA = bool(onnx_node.attrs.get('transA', 0))
    transB = bool(onnx_node.attrs.get('transB', 0))

    torch_module = TorchGemm(alpha, beta, transA, transB)
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
        'params': onnx_node.input_names[1:]
    }
    return torch_module, onnx_mapping
