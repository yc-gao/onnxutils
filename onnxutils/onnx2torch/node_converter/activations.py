from torch import nn

from onnxutils.onnx import OnnxModel, OnnxNode

from ..converter_registry import add_converter


@add_converter(op_type='Relu', version=14)
def _(onnx_node: OnnxNode, _: OnnxModel):
    torch_module = nn.ReLU()
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping


@add_converter(op_type='Sigmoid', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    torch_module = nn.Sigmoid()
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping


@add_converter(op_type='Softmax', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    axis = onnx_node.attributes().get('axis', -1)

    torch_module = nn.Softmax(axis)
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
