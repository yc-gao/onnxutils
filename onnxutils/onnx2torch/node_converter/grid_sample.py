from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchGridSample(nn.Module):
    def __init__(self, align_corners, mode, padding_mode):
        super().__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, x0, x1):
        return nn.functional.grid_sample(
            x0,
            x1,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners
        )


@add_converter(op_type='GridSample', version=16)
def _(onnx_node: OnnxNode, _: OnnxModel):
    align_corners = bool(onnx_node.attributes().get('align_corners', 0))
    mode = onnx_node.attributes().get('mode', 'bilinear')
    padding_mode = onnx_node.attributes().get('padding_mode', 'zeros')

    torch_module = TorchGridSample(align_corners, mode, padding_mode)
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
