import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchPermute(nn.Module):
    def __init__(self, perm):
        super().__init__()
        self.perm = perm

    def forward(self, x):
        return torch.permute(x, self.perm)


@add_converter(op_type='Transpose', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    perm = onnx_node.attributes().get('perm')

    torch_module = TorchPermute(perm),
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
