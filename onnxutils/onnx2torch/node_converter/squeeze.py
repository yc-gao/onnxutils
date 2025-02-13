import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchSqueeze(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return torch.squeeze(x, self.axis)


@add_converter(op_type='Squeeze', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    axis = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy().tolist()

    torch_module = TorchSqueeze(axis)
    onnx_mapping = {
        'inputs': onnx_node.inputs()[:1],
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
