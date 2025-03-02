import torch
from torch import nn
from onnxutils.onnx import OnnxNode, OnnxModel

from ..converter_registry import add_converter


class TorchConcat(nn.Module):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, *args):
        return torch.cat(args, self.axis)


@add_converter(op_type='Concat', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    axis: int = int(onnx_node.attrs['axis'])  # type: ignore

    torch_module = TorchConcat(axis)
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
