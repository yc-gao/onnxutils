from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import add_converter

op_mapping = {
    1: nn.AvgPool1d,
    2: nn.AvgPool2d,
    3: nn.AvgPool3d,
}


@add_converter(op_type='GlobalAveragePool', version=1)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    shape = tuple(x.dim_value
                  for x in onnx_model.get_vinfo_by_name(
                      onnx_node.inputs()[0]).type.tensor_type.shape.dim)

    torch_module = op_mapping[len(shape) - 2](kernel_size=shape[2:])
    onnx_mapping = {
        'inputs': onnx_node.inputs(),
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
