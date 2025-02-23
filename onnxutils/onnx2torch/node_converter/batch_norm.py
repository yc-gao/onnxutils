from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from ..converter_registry import add_converter

op_mapping = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
}


@add_converter(op_type='BatchNormalization', version=15)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    epsilon = onnx_node.attributes().get('epsilon', 1e-5)
    momentum = 1 - onnx_node.attributes().get('momentum', 0.9)
    training_mode = bool(onnx_node.attributes().get('training_mode', 0))

    assert not training_mode, 'not implement'

    scale = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_torch()
    bias = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[2]).to_torch()
    input_mean = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[3]).to_torch()
    input_var = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[4]).to_torch()

    x_rank = len(
        onnx_model.get_vinfo_by_name(
            onnx_node.inputs()[0]).type.tensor_type.shape.dim)

    torch_cls = op_mapping[x_rank - 2]
    torch_module = torch_cls(
        num_features=scale.size(0),
        eps=epsilon,
        momentum=momentum
    )

    torch_module.weight.data = scale
    torch_module.bias.data = bias
    torch_module.running_var.data = input_var
    torch_module.running_mean.data = input_mean

    onnx_mapping = {
        'inputs': onnx_node.inputs()[:1],
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
