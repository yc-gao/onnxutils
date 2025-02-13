from torch import nn
from torch.nn import functional as F


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter

func_mapping = {
    1: F.conv1d,
    2: F.conv2d,
    3: F.conv3d,
}


class TorchConv(nn.Module):
    def __init__(
            self,
            stride,
            padding,
            dilation,
            groups,
            func,
    ):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.f = func

    def forward(self, input, weight, bias=None):
        return self.f(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)


@add_converter(op_type='Conv', version=11)
def _(onnx_node: OnnxNode, _: OnnxModel):
    auto_pad = onnx_node.attributes().get('auto_pad', 'NOTSET')
    dilations = onnx_node.attributes().get('dilations')
    group = onnx_node.attributes().get('group', 1)
    kernel_shape = onnx_node.attributes().get('kernel_shape')
    pads = onnx_node.attributes().get('pads')
    strides = onnx_node.attributes().get('strides')

    assert auto_pad == 'NOTSET', "not implement"
    assert pads[:len(pads) // 2] == pads[len(pads) // 2:], "not implement"

    pads = pads[:len(pads) // 2]

    torch_module = TorchConv(
        strides,
        pads,
        dilations,
        group,
        func_mapping[len(kernel_shape)],
    )
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
        'params': onnx_node.inputs()[1:]
    }
    return torch_module, onnx_mapping
