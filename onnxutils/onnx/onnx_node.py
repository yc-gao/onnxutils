from types import MappingProxyType

import onnx
from onnx import NodeProto, AttributeProto

from .onnx_tensor import OnnxTensor


def _parse_attribute_value(attribute: AttributeProto):
    if attribute.HasField('i'):
        value = attribute.i
    elif attribute.HasField('f'):
        value = attribute.f
    elif attribute.HasField('s'):
        value = str(attribute.s, 'utf-8')
    elif attribute.HasField('t'):
        value = OnnxTensor(attribute.t)
    elif attribute.ints:
        value = list(attribute.ints)
    elif attribute.floats:
        value = list(attribute.floats)
    elif attribute.strings:
        value = [str(s, 'utf-8') for s in attribute.strings]
    elif attribute.tensors:
        value = [OnnxTensor(t) for t in attribute.tensors]
    else:
        value = attribute
    return value


class OnnxNode:
    def __init__(self, node_pb: NodeProto) -> None:
        self._proto = node_pb
        self._attrs = {
            attr.name: _parse_attribute_value(attr)
            for attr in self._proto.attribute
        }

    @property
    def proto(self):
        return self._proto

    @property
    def name(self):
        return self.proto.name

    @property
    def domain(self):
        return self.proto.domain

    @property
    def op_type(self):
        return self.proto.op_type

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(self.proto.input)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self.proto.output)

    @property
    def attrs(self):
        return MappingProxyType(self._attrs)
