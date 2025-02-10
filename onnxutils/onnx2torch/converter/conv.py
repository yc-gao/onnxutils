from torch import nn
from torch.nn import functional as F


from onnxutils.onnx import OnnxModel, OnnxNode

from ..registry import converter
from ..common import OnnxToTorchModule, OnnxMapping,  OperationConverterResult

func_mapping = {
    1: F.conv1d,
    2: F.conv2d,
    3: F.conv3d,
}


class TorchConv(nn.Module, OnnxToTorchModule):
    def __init__(
            self,
            dilations,
            group,
            kernel_shape,
            pads,
            strides,
            func,
    ):
        super().__init__()

        self.dilations = dilations
        self.group = group
        self.kernel_shape = kernel_shape
        self.pads = pads
        self.strides = strides
        self.f = func

    def forward(self, input, weight, bias=None):
        return self.f(input, weight, bias, self.strides, self.pads, self.dilations, self.group)


@converter(operation_type='Conv', version=11)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    dilations = onnx_node.attributes().get('dilations')
    group = onnx_node.attributes().get('group')
    kernel_shape = onnx_node.attributes().get('kernel_shape')
    pads = onnx_node.attributes().get('pads')
    strides = onnx_node.attributes().get('strides')

    auto_pad = onnx_node.attributes().get('auto_pad', 'NOTSET')

    assert auto_pad == 'NOTSET', "not implement"
    assert pads[:len(pads) // 2] == pads[len(pads) // 2:], "not implement"

    pads = pads[:len(pads) // 2]

    return OperationConverterResult(
        torch_module=TorchConv(dilations,
                               group,
                               kernel_shape,
                               pads,
                               strides,
                               func_mapping[len(kernel_shape)]),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        )
    )
