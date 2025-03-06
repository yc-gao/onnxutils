import pydash
import torch
from torch import nn
import onnx

from ...onnx import OnnxNode, OnnxModel
from ..converter_registry import add_converter

func_mapping = {
    'Add': torch.add,
    'Sub': torch.sub,
    'Mul': torch.mul,
    'Div': torch.div,
    'Div_int': lambda a, b: torch.div(a, b, rounding_mode='trunc'),
    'Pow': torch.pow,
    'Equal': torch.eq,
    'Greater': torch.gt,
    'Less': torch.lt,
    'LessOrEqual': torch.le,
    'And': torch.logical_and,
}


class TorchBinaryOp(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x0, x1):
        return self.f(x0, x1)


@add_converter(op_type='Equal', version=13)
@add_converter(op_type='Greater', version=13)
@add_converter(op_type='Less', version=13)
@add_converter(op_type='LessOrEqual', version=16)
@add_converter(op_type='Add', version=14)
@add_converter(op_type='Sub', version=14)
@add_converter(op_type='Mul', version=14)
@add_converter(op_type='Div', version=14)
@add_converter(op_type='Pow', version=15)
@add_converter(op_type='And', version=7)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel):
    op_type = onnx_node.op_type
    if op_type == 'Div':
        integer_types = (
            onnx.TensorProto.DataType.UINT4,
            onnx.TensorProto.DataType.INT4,
            onnx.TensorProto.DataType.UINT8,
            onnx.TensorProto.DataType.INT8,
            onnx.TensorProto.DataType.UINT16,
            onnx.TensorProto.DataType.INT16,
            onnx.TensorProto.DataType.UINT32,
            onnx.TensorProto.DataType.INT32,
            onnx.TensorProto.DataType.UINT64,
            onnx.TensorProto.DataType.INT64,
        )

        input_dtypes = tuple(
            pydash.get(
                onnx_model.get_info_by_name(x), '.type.tensor_type.elem_type')
            for x in onnx_node.input_names
        )
        if any(x is None for x in input_dtypes):
            import warnings
            warnings.warn(
                f'get data type for {onnx_node.name} failed, use fdiv')
        elif all(x in integer_types for x in input_dtypes):
            op_type = 'Div_int'

    torch_module = TorchBinaryOp(func_mapping[op_type])
    onnx_mapping = {
        'name': onnx_node.name,
        'inputs': onnx_node.input_names,
        'outputs': onnx_node.output_names,
    }
    return torch_module, onnx_mapping
