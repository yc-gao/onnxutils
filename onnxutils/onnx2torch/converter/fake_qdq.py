from torch.ao.quantization import HistogramObserver, PerChannelMinMaxObserver

from onnxutils.onnx import OnnxModel, OnnxNode
from onnxutils.quantization import FakeQuantize

from ..registry import converter
from ..common import OperationConverterResult, OnnxMapping


@converter(operation_type='FakeQdq', version=16)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    print(onnx_node.name())
    scale = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_torch()
    zp = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[2]).to_torch()

    observer_cls = HistogramObserver
    if scale.numel() > 1:
        observer_cls = PerChannelMinMaxObserver

    torch_module = FakeQuantize(observer_cls)
    torch_module.scale = scale
    torch_module.zero_point = zp

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs()[:1],
            outputs=onnx_node.outputs(),
        ),
    )
