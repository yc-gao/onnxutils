from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from ..registry import converter
from ..common import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchGemm(nn.Module, OnnxToTorchModule):
    def __init__(self, weight, bias, alpha, beta, transA):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.transA = transA

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        if self.transA:
            x = x.T
        return x @ self.weight * self.alpha + self.bias * self.beta


@converter(operation_type='Gemm', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    alpha = onnx_node.attributes().get('alpha', 1.0)
    beta = onnx_node.attributes().get('beta', 1.0)
    transA = bool(onnx_node.attributes().get('transA', 0))
    transB = bool(onnx_node.attributes().get('transB', 0))

    weight = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_torch()
    bias = None
    if len(onnx_node.inputs()) >= 3:
        bias = onnx_model.get_initializer_by_name(onnx_node.inputs()[2]).to_torch()

    if alpha == 1 and beta == 1 and transA == 0 and transB == 1:
        torch_module = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        torch_module.weight = nn.Parameter(weight)
        if bias is not None:
            torch_module.bias = nn.Parameter(bias)
        return OperationConverterResult(
            torch_module=torch_module,
            onnx_mapping=OnnxMapping(
                inputs=onnx_node.inputs()[:1],
                outputs=onnx_node.outputs(),
                params=onnx_node.inputs()[1:],
            ),
        )

    if transB:
        weight = weight.T

    return OperationConverterResult(
        torch_module=TorchGemm(weight, bias, alpha, beta, transA),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs()[:1],
            outputs=onnx_node.outputs(),
        ),
    )
