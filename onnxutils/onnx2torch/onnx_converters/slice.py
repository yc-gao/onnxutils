import torch
from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter


class TorchSliceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, starts, ends, axes, steps) -> torch.Tensor:
        slices = [slice(None)] * data.dim()
        if steps is None:
            steps = [1] * data.dim()
        for i, axis in enumerate(axes):
            slices[axis] = slice(starts[i], ends[i], steps[i])
        return data[slices]

    @staticmethod
    def symbolic(g, data, starts, ends, axes, steps):
        return g.op("Slice", data, starts, ends, axes, steps)


class TorchSlice(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data, starts, ends, axes, steps=None):
        return TorchSliceFunc.apply(data, starts, ends, axes, steps)


@add_converter(op_type='Slice', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    torch_module = TorchSlice()
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
