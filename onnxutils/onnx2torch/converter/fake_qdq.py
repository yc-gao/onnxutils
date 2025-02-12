from onnxutils.onnx import OnnxModel, OnnxNode
from onnxutils.quantization import FixedFakeQuantize

from ..registry import converter
from ..common import OperationConverterResult, OnnxMapping


@converter(operation_type='FakeQdq', version=16)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    scale = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_torch()
    zp = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[2]).to_torch()

    torch_module = FixedFakeQuantize(scale, zp)

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs()[:1],
            outputs=onnx_node.outputs(),
        ),
    )
