import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from ..converter_registry import add_converter


class TorchArgMax(nn.Module):
    def __init__(self, axis, keepdims):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return torch.argmax(x, self.axis, self.keepdims)


@add_converter(op_type='ArgMax', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    axis = onnx_node.attributes().get('axis', 0)
    keepdims = bool(onnx_node.attributes().get('keepdims', 1))
    select_last_index = bool(
        onnx_node.attributes().get('select_last_index', 0))

    assert not select_last_index, 'not implement'

    torch_module = TorchArgMax(axis, keepdims)
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
