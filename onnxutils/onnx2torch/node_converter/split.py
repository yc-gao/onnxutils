import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from ..converter_registry import add_converter


class TorchSplit(nn.Module):
    def __init__(self, axis, split):
        super().__init__()
        self.axis = axis
        self.split = split

    def forward(self, x):
        return torch.split(x, self.split, self.axis)


@add_converter(op_type='Split', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    axis = onnx_node.attributes().get('axis', 0)

    split = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy().tolist()

    torch_module = TorchSplit(axis, split)
    onnx_mapping = {
        'inputs': onnx_node.inputs()[:1],
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
