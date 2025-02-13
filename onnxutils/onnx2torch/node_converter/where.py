import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchWhere(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cond, x, y):
        return torch.where(cond, x, y)


@add_converter(op_type='Where', version=16)
def _(onnx_node: OnnxNode, _: OnnxModel):
    torch_module = TorchWhere()
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
