from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchPad(nn.Module):
    def __init__(self, mode, pads, constant_value):
        super().__init__()
        self.mode = mode
        self.pads = pads
        self.constant_value = constant_value

    def forward(self, x):
        return nn.functional.pad(x, self.pads, self.mode, self.constant_value)


@add_converter(op_type='Pad', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    mode = onnx_node.attributes().get('mode', 'constant')

    pads = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy()
    pads = pads.reshape((2, -1)).T[::-1].reshape(-1).tolist()

    constant = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[2])

    if constant is not None:
        constant = constant.to_numpy().item()
    if constant is None:
        constant = 0

    torch_module = TorchPad(mode, pads, constant)
    onnx_mapping = {
        'inputs': onnx_node.inputs()[:1],
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
