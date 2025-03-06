from typing import Optional

from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter

nn_mapping = {
    1: nn.MaxPool1d,
    2: nn.MaxPool2d,
    3: nn.MaxPool3d,
}


@add_converter(op_type='MaxPool', version=12)
def _(onnx_node: OnnxNode, _: OnnxModel):
    auto_pad = onnx_node.attrs.get('auto_pad', 'NOTSET')
    ceil_mode = bool(onnx_node.attrs.get('ceil_mode', 0))
    dilations = onnx_node.attrs.get('dilations', None)  # default to 1
    kernel_shape: list[int] = onnx_node.attrs['kernel_shape']
    pads: Optional[list[int]] = onnx_node.attrs.get(
        'pads', None)  # default to 0
    storage_order = onnx_node.attrs.get('storage_order', 0)  # default to 0
    strides = onnx_node.attrs.get('strides', None)  # default to 1

    if dilations is None:
        dilations = [1] * len(kernel_shape)
    if pads is None:
        pads = [0, 0] * len(kernel_shape)
    if strides is None:
        strides = [1] * len(kernel_shape)

    assert auto_pad == 'NOTSET', 'not implement'
    assert pads[:len(pads)//2] == pads[len(pads)//2:], 'not implement'
    assert storage_order == 0, "not implement"

    pads = pads[:len(pads)//2]

    torch_cls = nn_mapping[len(kernel_shape)]
    torch_module = torch_cls(
        kernel_shape,
        strides,
        pads,
        dilations,
        ceil_mode=ceil_mode
    )
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
