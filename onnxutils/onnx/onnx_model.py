from typing import Optional, Union

import os
import uuid
from collections import Counter

import onnx
from onnx import OperatorSetIdProto, ValueInfoProto, TensorProto, NodeProto, ModelProto

from .onnx_tensor import OnnxTensor
from .onnx_node import OnnxNode


class OnnxModel:
    @classmethod
    def from_file(cls, fpath: Union[str, os.PathLike]):
        return cls(onnx.load(fpath))

    def save(self, fpath: Union[str, os.PathLike]):
        onnx.save(self._proto, fpath)

    def __init__(self, model: ModelProto):
        self.reindex(model)

    def clone(self):
        t = ModelProto()
        t.CopyFrom(self._proto)
        return OnnxModel(t)

    def reindex(self, model: ModelProto):
        model = onnx.shape_inference.infer_shapes(model)
        self._proto: ModelProto = model

        self._nodes: tuple[OnnxNode, ...] = tuple(
            OnnxNode(x) for x in self._proto.graph.node
        )
        self._inputs: tuple[ValueInfoProto, ...] = tuple(
            x for x in self._proto.graph.input)
        self._outputs: tuple[ValueInfoProto, ...] = tuple(
            x for x in self._proto.graph.output)
        self._initializers: tuple[OnnxTensor, ...] = tuple(
            OnnxTensor(x)
            for x in self._proto.graph.initializer)

        self._name_to_node: dict[str, OnnxNode] = {
            x.name(): x for x in self._nodes
        }
        self._output_to_node: dict[str, OnnxNode] = {
            output: node for node in self._nodes for output in node.outputs()
        }

        self._name_to_initializer: dict[str, OnnxTensor] = {
            x.name(): x for x in self._initializers
        }

        self._name_to_input: dict[str, ValueInfoProto] = {
            x.name: x for x in self._inputs
        }
        self._name_to_output: dict[str, ValueInfoProto] = {
            x.name: x for x in self._outputs
        }
        self._name_to_vinfo: dict[str, ValueInfoProto] = {
            x.name: x for x in self._proto.graph.value_info
        }
        self._name_to_vinfo.update(self._name_to_input)
        self._name_to_vinfo.update(self._name_to_output)

        self._name_to_counter: Counter = Counter(
            [
                input_name
                for node in self._proto.graph.node
                for input_name in node.input
            ] +
            [x.name for x in self._proto.graph.output]
        )

    def proto(self) -> ModelProto:
        return self._proto

    def opsets(self) -> tuple[OperatorSetIdProto, ...]:
        return tuple(x for x in self._proto.opset_import)

    def inputs(self) -> tuple[ValueInfoProto, ...]:
        return self._inputs

    def input_names(self) -> tuple[str, ...]:
        return tuple(x.name for x in self._inputs)

    def get_input_by_name(self, name: str) -> ValueInfoProto:
        return self._name_to_input.get(name, None)

    def outputs(self) -> tuple[ValueInfoProto, ...]:
        return self._outputs

    def output_names(self) -> tuple[str, ...]:
        return tuple(x.name for x in self._outputs)

    def get_output_by_name(self, name) -> ValueInfoProto:
        return self._name_to_output.get(name, None)

    def initializers(self) -> tuple[OnnxTensor, ...]:
        return self._initializers

    def initializer_names(self) -> tuple[str, ...]:
        return tuple(x.name() for x in self._initializers)

    def get_initializer_by_name(self, name) -> Optional[OnnxTensor]:
        return self._name_to_initializer.get(name, None)

    def nodes(self) -> tuple[OnnxNode, ...]:
        return self._nodes

    def node_names(self) -> tuple[str, ...]:
        return tuple(x.name() for x in self._nodes)

    def get_node_by_name(self, name) -> Optional[OnnxNode]:
        return self._name_to_node.get(name, None)

    def get_node_by_output(self, output) -> Optional[OnnxNode]:
        return self._output_to_node.get(output, None)

    def get_vinfo_by_name(self, name) -> Optional[ValueInfoProto]:
        return self._name_to_vinfo.get(name, None)

    def get_counter_of_tensor(self, name: str) -> int:
        return self._name_to_counter[name]

    def get_counter_of_node(self, name_or_node) -> int:
        if isinstance(name_or_node, str):
            name_or_node = self._name_to_node.get(name_or_node, None)
        assert name_or_node
        return max(
            self._name_to_counter[output_value]
            for output_value in name_or_node.outputs())

    def topological_sort(self, is_deterministic=False):
        node_visited: set[str] = set()
        sorted_nodes: list[OnnxNode] = []

        def do_sort(arr):
            if not is_deterministic:
                return arr
            if not isinstance(arr, list):
                arr = [x for x in arr]
            arr.sort()
            return arr

        def dfs(node: Optional[OnnxNode]):
            if node is None:
                return
            if node.name() in node_visited:
                return
            node_visited.add(node.name())
            for input_name in node.inputs():
                dfs(self.get_node_by_output(input_name))
            sorted_nodes.append(node)

        for output_name in do_sort(self.output_names()):
            dfs(self.get_node_by_output(output_name))

        model = self._proto
        model.graph.ClearField("node")
        model.graph.node.extend([x.proto() for x in sorted_nodes])
        self.reindex(model)

    def extract(
            self,
            input_names: Union[set[str], tuple[str, ...], list[str]],
            output_names: Union[set[str], tuple[str, ...], list[str]]):
        input_names = set(input_names)
        output_names = set(output_names)

        tensor_visited: set[str] = set()
        node_visited: set[str] = set()

        nodes: list[NodeProto] = []
        inputs: list[ValueInfoProto] = []
        outputs: list[ValueInfoProto] = [
            self.get_vinfo_by_name(x) for x in output_names
        ]
        initializers: list[TensorProto] = []

        def dfs(output_name: str):
            if output_name in tensor_visited:
                return
            tensor_visited.add(output_name)
            if output_name in input_names:
                inputs.append(self.get_vinfo_by_name(output_name))
            elif input := self.get_input_by_name(output_name):
                inputs.append(input)
            elif initializer := self.get_initializer_by_name(output_name):
                initializers.append(initializer.proto())
            elif node := self.get_node_by_output(output_name):
                if node.name() not in node_visited:
                    for iname in node.inputs():
                        dfs(iname)
                    nodes.append(node.proto())
                    node_visited.add(node.name())
            elif not output_name:
                return
            else:
                raise RuntimeError(f"unmatched tensor {output_name}")

        for output_name in output_names:
            dfs(output_name)

        model_pb = self.clone().proto()
        model_pb.ClearField('graph')
        model_pb.graph.name = self._proto.graph.name
        model_pb.graph.node.extend(nodes)
        model_pb.graph.input.extend(inputs)
        model_pb.graph.output.extend(outputs)
        model_pb.graph.initializer.extend(initializers)
        return OnnxModel(model_pb)

    def session(self):
        class Session:
            def __init__(self, onnx_model: OnnxModel):
                self._onnx_model: OnnxModel = onnx_model

                self._counter: int = 0

                self._remap_node_inputs: dict[str, str] = {}

                self._initializers_to_remove: list[OnnxTensor] = []
                self._initializers_to_add: list[TensorProto] = []

                self._nodes_to_remove: list[OnnxNode] = []
                self._nodes_to_add: list[NodeProto] = []

                self._inputs_to_remove: list[ValueInfoProto] = []
                self._inputs_to_add: list[ValueInfoProto] = []

                self._outputs_to_remove: list[ValueInfoProto] = []
                self._outputs_to_add: list[ValueInfoProto] = []

            def unique_name(self):
                while True:
                    name = f"random_{uuid.uuid1()}_{self._counter}"
                    name = name.replace('-', '_')
                    self._counter += 1
                    if self._onnx_model.get_node_by_name(name):
                        continue
                    if self._onnx_model.get_vinfo_by_name(name):
                        continue
                    if self._onnx_model.get_initializer_by_name(name):
                        continue
                    return name

            def add_initializer(self, tensor: TensorProto):
                self._initializers_to_add.append(tensor)

            def add_initializers(self, tensors: list[TensorProto]):
                self._initializers_to_add.extend(tensors)

            def remove_initializer(self, tensor: OnnxTensor):
                self._initializers_to_remove.append(tensor)

            def add_node(self, node: NodeProto):
                self._nodes_to_add.append(node)

            def remove_node(self, node: OnnxNode):
                self._nodes_to_remove.append(node)

            def remove_nodes(self, nodes: list[OnnxNode]):
                self._nodes_to_remove.extend(nodes)

            def remap_node_inputs(self, remap):
                self._remap_node_inputs.update(remap)

            def remove_input(self, input):
                self._inputs_to_remove.append(input)

            def add_input(self, input):
                self._inputs_to_add.append(input)

            def remove_output(self, output):
                self._outputs_to_remove.append(output)

            def add_output(self, output):
                self._outputs_to_add.append(output)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                onnx_model = self._onnx_model.proto()
                for node in onnx_model.graph.node:
                    for idx in range(len(node.input)):
                        while True:
                            new_value = self._remap_node_inputs.get(
                                node.input[idx], None)
                            if new_value is None:
                                break
                            node.input[idx] = new_value

                for x in self._initializers_to_remove:
                    onnx_model.graph.initializer.remove(x.proto())
                for x in self._nodes_to_remove:
                    onnx_model.graph.node.remove(x.proto())
                for x in self._inputs_to_remove:
                    onnx_model.graph.input.remove(x)
                for x in self._outputs_to_remove:
                    onnx_model.graph.output.remove(x)

                onnx_model.graph.initializer.extend(
                    self._initializers_to_add)
                onnx_model.graph.node.extend(self._nodes_to_add)

                onnx_model.graph.input.extend(self._inputs_to_add)
                onnx_model.graph.output.extend(self._outputs_to_add)

                from onnx.utils import Extractor
                e = Extractor(onnx_model)
                new_model = e.extract_model(
                    [x.name for x in onnx_model.graph.input],
                    [x.name for x in onnx_model.graph.output])
                self._onnx_model.reindex(new_model)

        return Session(self)
