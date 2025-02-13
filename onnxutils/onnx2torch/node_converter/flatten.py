from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


@add_converter(op_type='Flatten', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    axis = onnx_node.attributes().get('axis', 1)

    torch_module = nn.Flatten(axis)
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
