import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchUnsqueeze(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return torch.unsqueeze(x, self.axis)


class TorchReshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, self.shape)


@add_converter(op_type='Unsqueeze', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    axis = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy()
    if axis.size == 1:
        axis = int(axis)

        torch_module = TorchUnsqueeze(axis)
        onnx_mapping = {
            'inputs': onnx_node.inputs()[:1],
            'outputs': onnx_node.outputs(),
        }
        return torch_module, onnx_mapping

    vinfo = onnx_model.get_vinfo_by_name(onnx_node.outputs()[0])
    shape = [x.dim_value if x.HasField(
        'dim_value') else -1 for x in vinfo.type.tensor_type.shape.dim]
    torch_module = TorchReshape(shape),
    onnx_mapping = {
        'inputs': onnx_node.inputs()[:1],
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
