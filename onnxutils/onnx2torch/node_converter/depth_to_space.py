from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from ..converter_registry import add_converter


@add_converter(op_type='DepthToSpace', version=13)
def _(onnx_node: OnnxNode, _: OnnxModel):
    blocksize = onnx_node.attributes().get('blocksize')
    mode = onnx_node.attributes().get('mode', 'CRD')

    assert mode == 'CRD', 'not implement'

    torch_module = nn.PixelShuffle(blocksize)
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
