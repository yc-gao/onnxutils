import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from ..converter_registry import add_converter


class TorchReshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, self.shape)


@add_converter(op_type='Reshape', version=14)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    allowzero = bool(onnx_node.attributes().get('allowzero', 0))

    assert not allowzero, 'not implement'

    shape = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy().tolist()

    torch_module = TorchReshape(shape)
    onnx_mapping = {
        'inputs': onnx_node.inputs()[:1],
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
