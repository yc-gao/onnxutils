from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchPad(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x, pad, constant_value=None):
        return nn.functional.pad(x, pad, self.mode, constant_value)


@add_converter(op_type='Pad', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    mode = onnx_node.attributes().get('mode', 'constant')

    torch_module = TorchPad(mode),
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
