import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchSliceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, starts, ends, axes, steps) -> torch.Tensor:
        slices = [slice(None)] * data.dim()
        for i, axis in enumerate(axes):
            start = starts[i]
            end = ends[i]
            step = steps[i]
            slices[axis] = slice(start, end, step)
        return data[slices]

    @staticmethod
    def symbolic(g: torch.Graph, data, starts, ends, axes, steps) -> torch.Value:
        return g.op("Slice", data, starts, ends, axes, steps)


class TorchSlice(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, starts, ends, axes, steps):
        return TorchSliceFunc.apply(data, starts, ends, axes, steps)


@add_converter(op_type='Slice', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    torch_module = TorchSlice()
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
