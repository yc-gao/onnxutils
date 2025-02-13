import torch

from onnxutils.onnx import OnnxModel, OnnxNode
from onnxutils.quantization import FixedFakeQuantize

from .registry import add_converter

range_mapping = {
    torch.uint8: {
        'quant_min': 0,
        'quant_max': 255,
    }
}


@add_converter(op_type='FakeQdq', version=16)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    scale = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_torch()
    zp = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[2]).to_torch()

    torch_module = FixedFakeQuantize(scale, zp, **range_mapping[zp.dtype])

    torch_module = torch_module,
    onnx_mapping = {
        'inputs': onnx_node.inputs()[:1],
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
