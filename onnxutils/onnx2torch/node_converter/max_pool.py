from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from ..converter_registry import add_converter

op_mapping = {
    1: nn.MaxPool1d,
    2: nn.MaxPool2d,
    3: nn.MaxPool3d,
}


@add_converter(op_type='MaxPool', version=12)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):  # pylint: disable=unused-argument
    auto_pad = onnx_node.attributes().get('auto_pad', 'NOTSET')
    ceil_mode = bool(onnx_node.attributes().get('ceil_mode', 0))
    dilations = onnx_node.attributes().get('dilations', 1)
    kernel_shape = onnx_node.attributes().get('kernel_shape')
    pads = onnx_node.attributes().get('pads')
    storage_order = onnx_node.attributes().get('storage_order', 0)
    strides = onnx_node.attributes().get('strides', 1)

    assert auto_pad == "NOTSET", "not implement"
    assert pads[:len(pads) // 2] == pads[len(pads) // 2:], "not implement"
    assert storage_order == 0, "not implement"

    torch_cls = op_mapping[len(kernel_shape)]
    torch_module = torch_cls(
        kernel_shape,
        strides,
        pads[:len(pads)//2],
        dilations,
        ceil_mode=ceil_mode
    )
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
