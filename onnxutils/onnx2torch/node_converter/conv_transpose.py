from torch import nn
from torch.nn import functional as F


from onnxutils.onnx import OnnxModel, OnnxNode

from ..converter_registry import add_converter

func_mapping = {
    1: F.conv_transpose1d,
    2: F.conv_transpose2d,
    3: F.conv_transpose3d,
}


class TorchConvTranspose(nn.Module):
    def __init__(
        self,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        func,
    ):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

        self.f = func

    def forward(self, input, weight, bias=None):
        return self.f(input, weight, bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)


@add_converter(op_type='ConvTranspose', version=11)
def _(onnx_node: OnnxNode, _: OnnxModel):
    auto_pad = onnx_node.attributes().get('auto_pad', 'NOTSET')
    dilations = onnx_node.attributes().get('dilations')
    group = onnx_node.attributes().get('group', 1)
    kernel_shape = onnx_node.attributes().get('kernel_shape')
    output_padding = onnx_node.attributes().get('output_padding', 0)
    output_shape = onnx_node.attributes().get('output_shape')
    pads = onnx_node.attributes().get('pads')
    strides = onnx_node.attributes().get('strides')

    assert output_shape is None, 'not implement'
    assert auto_pad == 'NOTSET', "not implement"
    assert pads[:len(pads) // 2] == pads[len(pads) // 2:], "not implement"

    pads = pads[:len(pads) // 2]

    torch_module = TorchConvTranspose(
        strides,
        pads,
        output_padding,
        group,
        dilations,
        func_mapping[len(kernel_shape)])
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
        'params': onnx_node.inputs()[1:]
    }
    return torch_module, onnx_mapping
