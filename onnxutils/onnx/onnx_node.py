from types import MappingProxyType

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
    def __init__(self, onnx_node: NodeProto):
        self._proto: NodeProto = onnx_node

        self._inputs: tuple[str, ...] = tuple(self._proto.input)
        self._outputs: tuple[str, ...] = tuple(self._proto.output)
        self._attrs: dict[str, object] = {
            attr.name: _parse_attribute_value(attr)
            for attr in self._proto.attribute
        }

    def clone(self):
        t = NodeProto()
        t.CopyFrom(self._proto)
        return OnnxNode(t)

    def proto(self) -> NodeProto:
        return self._proto

    def name(self) -> str:
        return self._proto.name

    def domain(self) -> str:
        return self._proto.domain

    def op_type(self) -> str:
        return self._proto.op_type

    def inputs(self) -> tuple[str, ...]:
        return self._inputs

    def outputs(self) -> tuple[str, ...]:
        return self._outputs

    def attributes(self) -> MappingProxyType[str, object]:
        return MappingProxyType(self._attrs)

    def attribute(self, name, val=None):
        return self._attrs.get(name, val)
