from torch import nn
from torch.nn import functional as F


from onnxutils.onnx import OnnxModel, OnnxNode

from ..registry import converter
from ..common import OnnxMapping,  OperationConverterResult

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
        groups,
        dilation,
        func,
    ):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        self.f = func

    def forward(self, input, weight, bias):
        return self.f(input, weight, bias, self.stride, self.padding, 0, self.groups, self.dilation)


@converter(operation_type='ConvTranspose', version=11)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    dilations = onnx_node.attributes().get('dilations')
    group = onnx_node.attributes().get('group')
    kernel_shape = onnx_node.attributes().get('kernel_shape')
    pads = onnx_node.attributes().get('pads')
    strides = onnx_node.attributes().get('strides')

    output_padding = onnx_node.attributes().get('output_padding')
    output_shape = onnx_node.attributes().get('output_shape')
    auto_pad = onnx_node.attributes().get('auto_pad', 'NOTSET')

    assert output_padding is None, 'not implement'
    assert output_shape is None, 'not implement'
    assert auto_pad == 'NOTSET', "not implement"
    assert pads[:len(pads) // 2] == pads[len(pads) // 2:], "not implement"

    pads = pads[:len(pads) // 2]

    return OperationConverterResult(
        torch_module=TorchConvTranspose(
            strides,
            pads,
            group,
            dilations,
            func_mapping[len(kernel_shape)]),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        )
    )
