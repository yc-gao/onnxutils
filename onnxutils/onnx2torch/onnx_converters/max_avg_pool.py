import pydash
from torch import nn

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter

nn_mapping = {
    1: nn.AdaptiveAvgPool1d,
    2: nn.AdaptiveAvgPool2d,
    3: nn.AdaptiveAvgPool3d,
}


@add_converter(op_type='GlobalAveragePool', version=1)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    shape = pydash.get(onnx_model.get_info_by_name(
        onnx_node.output_names[0]),
        'type.tensor_type.shape.dim'
    )
    shape = shape and tuple(x.dim_value for x in shape)

    torch_module = nn_mapping[len(shape) - 2](output_size=shape[2:])
    onnx_mapping = {
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
