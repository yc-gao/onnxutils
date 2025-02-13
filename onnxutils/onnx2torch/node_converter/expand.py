from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter


class TorchExpand(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.expand(*self.shape)


@add_converter(op_type='Expand', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    vinfo = onnx_model.get_vinfo_by_name(onnx_node.outputs()[0])
    shape = [x.dim_value if x.HasField(
        'dim_value') else -1 for x in vinfo.type.tensor_type.shape.dim]

    torch_module = TorchExpand(shape)
    onnx_mapping = {
        'inputs': onnx_node.inputs()[:1],
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
