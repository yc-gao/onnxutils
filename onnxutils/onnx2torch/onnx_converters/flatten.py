from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter


@add_converter(op_type='Flatten', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    axis: int = onnx_node.attrs.get('axis', 1)

    torch_module = nn.Flatten(axis)
    onnx_mapping = {
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
