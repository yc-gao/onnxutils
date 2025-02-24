from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from ..converter_registry import add_converter


class TorchGather(nn.Module):
    def __init__(self, axis, index):
        super().__init__()
        self.axis = axis
        self.index = index

    def forward(self, x):
        axis = self.axis
        if axis < 0:
            axis += x.dim()

        if axis == 0:
            return x[self.index, ...]
        elif axis == 1:
            return x[:, self.index, ...]
        elif axis == 2:
            return x[:, :, self.index, ...]
        elif axis == 3:
            return x[:, :, :, self.index, ...]
        elif axis == 4:
            return x[:, :, :, :, self.index, ...]
        else:
            raise NotImplementedError


@add_converter(op_type='Gather', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    axis = onnx_node.attributes().get('axis', 0)
    index = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy().tolist()

    torch_module = TorchGather(axis, index)
    onnx_mapping = {
        'inputs': onnx_node.inputs()[:1],
        'outputs': onnx_node.outputs(),
    }
    return torch_module, onnx_mapping
