import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from ..converter_registry import add_converter


class TorchReduceMean(nn.Module):
    def __init__(self, axis, keepdims):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return torch.mean(x, dim=self.axis, keepdim=self.keepdims)


@add_converter(op_type='ReduceMean', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    axis = onnx_node.attributes().get('axes')
    keepdims = bool(onnx_node.attributes().get('keepdims', 1))

    assert len(axis) == 1, 'not implement'
    axis = axis[0]

    torch_module = TorchReduceMean(axis, keepdims)
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
