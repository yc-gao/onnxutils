import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchConcat(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, *args):
        return torch.cat(args, self.axis)


@add_converter(op_type='Concat', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    axis = onnx_node.attributes().get('axis')
    torch_module = TorchConcat(axis),
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
