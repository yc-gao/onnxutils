import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchGatherElements(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x, index):
        return torch.gather(x, self.axis, index)


@add_converter(op_type='GatherElements', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    axis = onnx_node.attributes().get('axis', 0)

    torch_module = TorchGatherElements(axis),
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
