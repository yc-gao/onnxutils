import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchClip(nn.Module):
    def __init__(self, min_val=None, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clip(x, self.min_val, self.max_val)


@add_converter(op_type='Clip', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    min_val = onnx_model.get_initializer_by_name(onnx_node.inputs()[1])
    max_val = onnx_model.get_initializer_by_name(onnx_node.inputs()[2])

    if min_val is not None:
        min_val = min_val.to_numpy().item()
    if max_val is not None:
        max_val = max_val.to_numpy().item()

    torch_module = TorchClip(min_val, max_val),
    onnx_mapping = {
        'inputs': onnx_node.inputs()[:1],
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
