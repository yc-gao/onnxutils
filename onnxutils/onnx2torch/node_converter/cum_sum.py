import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from ..converter_registry import add_converter


class TorchCumSum(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return torch.cumsum(x, self.axis)


@add_converter(op_type='CumSum', version=14)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    exclusive = onnx_node.attributes().get('exclusive', 0)
    reverse = onnx_node.attributes().get('reverse', 0)

    assert exclusive == 0, 'not implement'
    assert reverse == 0, 'not implement'

    axis = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy().item()

    torch_module = TorchCumSum(axis)
    onnx_mapping = {
        'inputs': onnx_node.inputs()[:1],
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
