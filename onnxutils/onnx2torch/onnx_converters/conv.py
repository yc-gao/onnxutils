from typing import Optional
from torch import nn
from torch.nn import functional as F
from onnxutils.onnx import OnnxNode, OnnxModel

from ..converter_registry import add_converter

func_mapping = {
    1: F.conv1d,
    2: F.conv2d,
    3: F.conv3d,
}


class TorchConv(nn.Module):
    def __init__(self, f, stride, padding, dilation, groups) -> None:
        super().__init__()
        self.f = f
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input, weight, bias=None):
        self.f(
            input, weight, bias,
            self.stride, self.padding, self.dilation, self.groups
        )


@add_converter(op_type='Conv', version=11)
def _(onnx_node: OnnxNode, _: OnnxModel):
    auto_pad = onnx_node.attrs.get('auto_pad', 'NOTSET')
    dilations = onnx_node.attrs.get('dilations', None)  # default to 1
    group: int = onnx_node.attrs.get('group', 1)
    kernel_shape: list[int] = onnx_node.attrs['kernel_shape']
    pads: Optional[list[int]] = onnx_node.attrs.get(
        'pads', None)  # default to 0
    strides = onnx_node.attrs.get('strides', None)  # default to 1

    if dilations is None:
        dilations = [1] * len(kernel_shape)
    if pads is None:
        pads = [0, 0] * len(kernel_shape)
    if strides is None:
        strides = [1] * len(kernel_shape)

    assert auto_pad == 'NOTSET', 'not implement'
    assert pads[:len(pads)//2] == pads[len(pads)//2:], 'not implement'
    pads = pads[:len(pads)//2]

    torch_module = TorchConv(
        func_mapping[len(kernel_shape)],
        strides,
        pads,
        dilations,
        group
    )
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
