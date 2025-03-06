import torch
from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter


class TorchGatherFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, indices, axis) -> torch.Tensor:
        slices = [slice(None)] * data.dim()
        slices[axis] = indices
        return data[slices]

    @staticmethod
    def symbolic(g, data, indices, axis):
        return g.op("Gather", data, indices, axis_i=axis)


class TorchGather(nn.Module):
    def __init__(self, axis) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, data, indices):
        return TorchGatherFunc.apply(data, indices, self.axis)


@add_converter(op_type='Gather', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    axis: int = onnx_node.attrs.get('axis', 0)

    torch_module = TorchGather(axis)
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
