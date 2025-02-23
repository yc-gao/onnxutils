from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from ..converter_registry import add_converter


class TorchGemm(nn.Module):
    def __init__(self, alpha, beta, transA, transB):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB

    def forward(self, x, weight, bias):
        if self.transA:
            x = x.T
        if self.transB:
            weight = weight.T

        return x @ weight * self.alpha + bias * self.beta


@add_converter(op_type='Gemm', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    alpha = onnx_node.attributes().get('alpha', 1.0)
    beta = onnx_node.attributes().get('beta', 1.0)
    transA = bool(onnx_node.attributes().get('transA', 0))
    transB = bool(onnx_node.attributes().get('transB', 0))

    if alpha == 1 and beta == 1 and not transA and transB:
        weight = onnx_model.get_initializer_by_name(
            onnx_node.inputs()[1]).to_torch()
        bias = None
        if len(onnx_node.inputs()) > 2:
            bias = onnx_model.get_initializer_by_name(
                onnx_node.inputs()[2]).to_torch()

        torch_module = nn.Linear(
            weight.shape[1], weight.shape[0], bias=bias is not None)
        torch_module.weight = nn.Parameter(weight)
        if bias is not None:
            torch_module.bias = nn.Parameter(bias)
        onnx_mapping = {
            'inputs': onnx_node.inputs()[:1],
            'outputs': onnx_node.outputs(),
            'params': onnx_node.inputs()[1:],
        }
        return torch_module, onnx_mapping

    torch_module = TorchGemm(alpha, beta, transA, transB)
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
        'params': onnx_node.inputs()[1:],
    }
    return torch_module, onnx_mapping
