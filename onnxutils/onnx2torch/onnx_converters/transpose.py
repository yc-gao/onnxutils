import torch
from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter


class TorchTranspose(nn.Module):
    def __init__(self, perm) -> None:
        super().__init__()
        self.perm = perm

    def forward(self, data):
        return torch.permute(data, self.perm)


@add_converter(op_type='Transpose', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    perm: list[int] = onnx_node.attrs['perm']

    torch_module = TorchTranspose(perm)
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
