from typing import Optional, Union

import os
import copy
import uuid
from contextlib import contextmanager

import onnx
from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto, OperatorSetIdProto

from .onnx_tensor import OnnxTensor
from .onnx_node import OnnxNode


class OnnxModel:
    @staticmethod
    def from_file(fpath: Union[str, os.PathLike]):
        return OnnxModel(onnx.load(fpath))

    def __init__(self, model_pb: ModelProto) -> None:
        self.refresh(model_pb)

    def refresh(self, model_pb: Optional[ModelProto] = None):
        if model_pb is None:
            model_pb = self._proto

        assert model_pb
        model_pb = onnx.shape_inference.infer_shapes(model_pb)

        self._proto = model_pb
        self._tensors = tuple(
            OnnxTensor(x)
            for x in self._proto.graph.initializer)
        self._nodes = tuple(
            OnnxNode(x)
            for x in self._proto.graph.node
        )

        self._name2node = {
            x.name: x for x in self._nodes
        }
        self._output2node = {
            output: x for x in self._nodes for output in x.output_names
        }

        self._name2tensor = {
            x.name: x for x in self._tensors
        }

        self._name2vinfo = {
            x.name: x for x in self._proto.graph.value_info
        }
        self._name2vinfo.update({
            x.name: x for x in self._proto.graph.input
        })
        self._name2vinfo.update({
            x.name: x for x in self._proto.graph.output
        })

    @property
    def proto(self):
        return self._proto

    @property
    def opsets(self) -> tuple[OperatorSetIdProto, ...]:
        return tuple(x for x in self.proto.opset_import)

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(x.name for x in self.proto.graph.input)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(x.name for x in self.proto.graph.output)

    @property
    def initializers(self) -> tuple[OnnxTensor, ...]:
        return self._tensors

    @property
    def initializer_names(self) -> tuple[str, ...]:
        return tuple(x.name for x in self.initializers)

    @property
    def nodes(self) -> tuple[OnnxNode, ...]:
        return self._nodes

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(x.name for x in self.nodes)

    def get_node_by_name(self, name) -> Optional[OnnxNode]:
        return self._name2node.get(name, None)

    def get_node_by_output(self, output_name) -> Optional[OnnxNode]:
        return self._output2node.get(output_name, None)

    def get_initializer_by_name(self, name) -> Optional[OnnxTensor]:
        return self._name2tensor.get(name, None)

    def get_info_by_name(self, name) -> Optional[ValueInfoProto]:
        return self._name2vinfo.get(name, None)

    def topological_sort(self):
        node_visited = set()

        sorted_nodes = []

        def dfs(tensor_name):
            onnx_node = self.get_node_by_output(tensor_name)
            if onnx_node is None:
                return
            if onnx_node.name in node_visited:
                return
            node_visited.add(onnx_node.name)

            for t in onnx_node.input_names:
                dfs(t)
            sorted_nodes.append(onnx_node.proto)

        for output_name in self.output_names:
            dfs(output_name)

        self.proto.graph.ClearField('node')
        self.proto.graph.node.extend(sorted_nodes)
        self.refresh()

    def extract(self,
                input_names: Union[set[str], list[str], tuple[str, ...]],
                output_names: Union[set[str], list[str], tuple[str, ...]]):
        input_names = set(input_names)
        output_names = set(output_names)

        tensor_visited = set()
        node_visited = set()

        inputs: list[ValueInfoProto] = []
        outputs: list[ValueInfoProto] = [
            self._name2vinfo[x]
            for x in output_names
        ]
        initializers: list[TensorProto] = []
        nodes: list[NodeProto] = []

        def dfs(tensor_name):
            if tensor_name in tensor_visited:
                return
            tensor_visited.add(tensor_name)

            if onnx_node := self.get_node_by_output(tensor_name):
                if onnx_node.name in node_visited:
                    return
                node_visited.add(onnx_node.name)
                for input_name in onnx_node.input_names:
                    dfs(input_name)
                nodes.append(onnx_node.proto)
            elif initializer := self.get_initializer_by_name(tensor_name):
                initializers.append(initializer.proto)
            elif tensor_name in input_names:
                inputs.append(self._name2vinfo[tensor_name])
            elif tensor_name in self.input_names:
                inputs.append(self._name2vinfo[tensor_name])
            else:
                assert not tensor_name, f"unmatched tensor '{tensor_name}'"

        for output_name in output_names:
            dfs(output_name)

        model_pb = copy.deepcopy(self.proto)
        model_pb.ClearField('graph')
        model_pb.graph.name = self._proto.graph.name
        model_pb.graph.input.extend(inputs)
        model_pb.graph.output.extend(outputs)
        model_pb.graph.initializer.extend(initializers)
        model_pb.graph.node.extend(nodes)
        return OnnxModel(model_pb)

    @contextmanager
    def modify(self):
        class Session:
            def __init__(self, onnx_model: OnnxModel) -> None:
                self._onnx_model = onnx_model
                self._remap_node_inputs: dict[str, str] = {}

                self._initializers_to_remove: list[OnnxTensor] = []
                self._initializers_to_add: list[TensorProto] = []

                self._inputs_to_remove: list[ValueInfoProto] = []
                self._inputs_to_add: list[ValueInfoProto] = []

                self._outputs_to_remove: list[ValueInfoProto] = []
                self._outputs_to_add: list[ValueInfoProto] = []

                self._nodes_to_remove: list[OnnxNode] = []
                self._nodes_to_add: list[NodeProto] = []

                self._counter: int = 0

            def unique_name(self):
                while True:
                    name = f"random_{uuid.uuid1()}_{self._counter}"
                    name = name.replace('-', '_')
                    self._counter += 1
                    if self._onnx_model.get_node_by_name(name):
                        continue
                    if self._onnx_model.get_initializer_by_name(name):
                        continue
                    if self._onnx_model.get_info_by_name(name):
                        continue
                    return name

            def remap_node_inputs(self, remap):
                self._remap_node_inputs.update(remap)

            def add_initializer(self, initializer):
                self._initializers_to_add.append(initializer)

            def remove_initializer(self, initializer):
                self._initializers_to_remove.append(initializer)

            def add_input(self, x):
                self._inputs_to_add.append(x)

            def remove_input(self, x):
                self._inputs_to_remove.append(x)

            def add_output(self, x):
                self._outputs_to_add.append(x)

            def remove_output(self, x):
                self._outputs_to_remove.append(x)

            def add_node(self, node):
                self._nodes_to_add.append(node)

            def remove_node(self, node):
                self._nodes_to_remove.append(node)

            def finalize(self):
                model_pb: ModelProto = self._onnx_model.proto
                for node in model_pb.graph.node:
                    for idx, _ in enumerate(node.input):
                        while True:
                            new_value = self._remap_node_inputs.get(
                                node.input[idx], None)
                            if new_value is None:
                                break
                            node.input[idx] = new_value

                for x in self._initializers_to_remove:
                    model_pb.graph.initializer.remove(x.proto)
                model_pb.graph.initializer.extend(self._initializers_to_add)

                for x in self._nodes_to_remove:
                    model_pb.graph.node.remove(x.proto)
                model_pb.graph.node.extend(self._nodes_to_add)

                for x in self._inputs_to_remove:
                    model_pb.graph.input.remove(x)
                model_pb.graph.input.extend(self._inputs_to_add)

                for x in self._outputs_to_remove:
                    model_pb.graph.input.remove(x)
                model_pb.graph.input.extend(self._outputs_to_add)

                self._onnx_model.refresh(model_pb)

        sess = Session(self)
        yield sess
        sess.finalize()
