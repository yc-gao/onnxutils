from typing import Union

from operator import getitem

import torch
from onnx import OperatorSetIdProto

from onnxutils.onnx import OnnxModel

from .converter_registry import find_converter


def normalize_module_name(
        name: str,
        domain: str = '',
        op_type: str = '') -> str:
    name = name or f'{domain}.{op_type}'
    name = name.replace('.', '_')
    name = name.replace('/', '_')
    return name


def convert(
    onnx_model: OnnxModel,
    keep_input_names: bool = True,
) -> torch.fx.GraphModule:

    opset_import: dict[str, OperatorSetIdProto] = {
        opsetid_proto.domain: opsetid_proto.version for opsetid_proto in onnx_model.opsets()  # noqa
    }

    root_module: torch.nn.Module = torch.nn.Module()
    root_graph: torch.fx.Graph = torch.fx.Graph()

    nodes_mapping: dict[
        Union[str, tuple[torch.fx.Node, int]],
        torch.fx.Node
    ] = {}

    def get_or_create_torch_node(value_name):
        if onnx_model.get_input_by_name(value_name):
            return nodes_mapping[value_name]

        elif initializer_value := onnx_model.get_initializer_by_name(value_name):  # noqa
            torch_node = nodes_mapping.get(value_name, None)
            if torch_node is not None:
                return torch_node

            initializer_value = initializer_value.to_torch()
            initializer_value.onnx_mapping = {
                'name': value_name,
            }
            buffer_count = sum(1 for _ in root_module.buffers())
            buffer_name = f'onnx_initializer_{buffer_count}'
            root_module.register_buffer(buffer_name, initializer_value)
            torch_node = root_graph.get_attr(buffer_name)
            nodes_mapping[value_name] = torch_node
            return torch_node

        elif onnx_node := onnx_model.get_node_by_output(value_name):
            torch_node = nodes_mapping[onnx_node.name()]
            if len(onnx_node.outputs()) <= 1:
                return torch_node
            index = onnx_node.outputs().index(value_name)
            maybe_torch_node = nodes_mapping.get(
                (torch_node, index),
                None
            )
            if maybe_torch_node is not None:
                return maybe_torch_node
            maybe_torch_node = root_graph.call_function(
                getitem,
                args=(torch_node, index)
            )
            nodes_mapping[(torch_node, index)] = maybe_torch_node
            return maybe_torch_node
        else:
            raise RuntimeError(
                f"got unexpected value name '{value_name}'")

    # create input nodes
    for idx, name in enumerate(onnx_model.input_names()):
        if keep_input_names:
            assert name.isidentifier(), f"input name '{name}' cannot be used as placeholder name"  # noqa
            placeholder_name = name
        else:
            placeholder_name = f'input_{idx}'
        nodes_mapping[name] = root_graph.placeholder(name=placeholder_name)

    for onnx_node in onnx_model.nodes():
        converter = find_converter(
            op_type=onnx_node.op_type(),
            version=opset_import[onnx_node.domain()],
            domain=onnx_node.domain(),
        )

        torch_module, onnx_mapping = converter(onnx_node, onnx_model)
        torch_module.onnx_mapping = onnx_mapping

        root_module.add_module(
            normalize_module_name(onnx_node.name()),
            torch_module
        )
        nodes_mapping[onnx_node.name()] = root_graph.call_module(
            module_name=normalize_module_name(onnx_node.name()),
            args=tuple(
                get_or_create_torch_node(x)
                for x in onnx_mapping.get('inputs', [])
            )
        )

    outputs = tuple(get_or_create_torch_node(x)
                    for x in onnx_model.output_names())
    if len(outputs) > 1:
        root_graph.output(outputs)
    elif len(outputs) == 1:
        root_graph.output(outputs[0])
    else:
        raise RuntimeError("got no val to return")

    root_graph.lint()
    torch_model = torch.fx.GraphModule(root=root_module, graph=root_graph)
    torch_model.onnx_mapping = {
        'inputs': onnx_model.input_names(),
        'outputs': onnx_model.output_names(),
    }
    return torch_model
