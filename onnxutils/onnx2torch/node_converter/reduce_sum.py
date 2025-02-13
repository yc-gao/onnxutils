import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchReduceSum(nn.Module):
    def __init__(self, axis, keepdims, noop_with_empty_axes):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        self.noop_with_empty_axes = noop_with_empty_axes

    def forward(self, x):
        if self.axis is None:
            if self.noop_with_empty_axes:
                return x
            else:
                return torch.sum(x)
        else:
            return torch.sum(x, dim=self.axis, keepdim=self.keepdims)


@add_converter(op_type='ReduceSum', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    keepdims = bool(onnx_node.attributes().get('keepdims', 1))
    noop_with_empty_axes = bool(
        onnx_node.attributes().get('noop_with_empty_axes', 0))

    axis = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]) if len(onnx_node.inputs()) > 1 else None

    if axis is not None:
        axis = axis.to_numpy().tolist()

    torch_module = TorchReduceSum(axis, keepdims, noop_with_empty_axes)
    onnx_mapping = {
        'inputs': onnx_node.inputs()[:1],
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
