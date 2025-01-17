from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OperationConverterResult, OnnxMapping


@converter(operation_type='Relu', version=14)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    return OperationConverterResult(
        torch_module=nn.ReLU(),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        ),
    )


@converter(operation_type='Sigmoid', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    return OperationConverterResult(
        torch_module=nn.Sigmoid(),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        ),
    )
