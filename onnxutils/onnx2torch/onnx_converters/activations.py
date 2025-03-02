from torch import nn
from onnxutils.onnx import OnnxNode, OnnxModel

from ..converter_registry import add_converter


@add_converter(op_type='Relu', version=14)
def _(onnx_node: OnnxNode, _: OnnxModel):
    torch_module = nn.ReLU()
    onnx_mapping = {
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping


@add_converter(op_type='Sigmoid', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    torch_module = nn.Sigmoid()
    onnx_mapping = {
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
