from operator import getitem
import warnings

import torch

from onnxutils.onnx import OnnxModel

from .node_converter.registry import find_converter


class InitializersContainer(torch.nn.Module):
    def add_initializer(self, name: str, initializer: torch.Tensor) -> None:
        self.register_buffer(name, initializer)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def normalize_module_name(name, domain='', op_type=''):
    name = name or op_type
    name = name.replace('.', '_')
    name = name.replace('/', '_')
    return name


def convert(
    onnx_model: OnnxModel,
    keep_input_names: bool = True,
) -> torch.fx.GraphModule:
    opset_import = {
        opsetid_proto.domain: opsetid_proto.version for opsetid_proto in onnx_model.opsets()}

    root_initializer = InitializersContainer()

    root_module = torch.nn.Module()
    root_module.add_module('initializers', root_initializer)

    torch_graph = torch.fx.Graph()

    torch_nodes = {}

    # create input nodes
    for idx, name in enumerate(onnx_model.input_names()):
        if keep_input_names:
            if not name.isidentifier():
                raise ValueError(
                    f"Input name '{name}' cannot be used as name of placeholder in fx.GraphModule.")
            placeholder_name = name
        else:
            placeholder_name = f'input_{idx}'
        torch_nodes[name] = torch_graph.placeholder(name=placeholder_name)

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

        args = []
        for value_name in onnx_mapping.get('inputs', []):
            if onnx_model.get_input_by_name(value_name) is not None:
                args.append(torch_nodes[value_name])
            elif onnx_model.get_initializer_by_name(value_name) is not None:
                if value_name not in torch_nodes:
                    initializer_value = onnx_model.get_initializer_by_name(
                        value_name).to_torch()
                    initializer_value.onnx_mapping = {
                        'name': value_name,
                        'is_parameter': value_name in onnx_mapping.get('params', [])
                    }
                    buffer_idx = sum(
                        1 for _ in root_initializer.buffers())
                    buffer_name = f'onnx_initializer_{buffer_idx}'
                    root_initializer.add_initializer(
                        buffer_name,
                        initializer_value,
                    )
                    torch_nodes[value_name] = torch_graph.get_attr(
                        f'initializers.{buffer_name}')
                args.append(torch_nodes[value_name])
            elif onnx_input_node := onnx_model.get_node_by_output(value_name):
                torch_input_node = torch_nodes[onnx_input_node.name()]
                if len(onnx_input_node.outputs()) > 1:
                    index = onnx_input_node.outputs().index(value_name)
                    maybe_torch_input_node = torch_nodes.get(
                        (torch_input_node, index),
                        None)
                    if maybe_torch_input_node is None:
                        maybe_torch_input_node = torch_graph.call_function(
                            getitem, args=(torch_input_node, index))
                        torch_nodes[(torch_input_node, index)] = maybe_torch_input_node
                    torch_input_node = maybe_torch_input_node
                args.append(torch_input_node)
            else:
                warnings.warn(
                    f'Got unexpected input value name ({value_name})')

        torch_nodes[onnx_node.name()] = torch_graph.call_module(
            module_name=normalize_module_name(onnx_node.name()), args=tuple(args))

    args = []
    for output_name in onnx_model.output_names():
        onnx_output_node = onnx_model.get_node_by_output(output_name)
        assert onnx_output_node is not None
        torch_output_node = torch_nodes[onnx_output_node.name()]
        if len(onnx_output_node.outputs()) > 1:
            index = onnx_output_node.outputs().index(output_name)
            torch_output_node = torch_graph.call_function(
                getitem, args=(torch_output_node, index))
        args.append(torch_output_node)

    if len(args) > 1:
        torch_graph.output(tuple(args))
    else:
        torch_graph.output(args[0])

    torch_graph.lint()

    torch_model = torch.fx.GraphModule(root=root_module, graph=torch_graph)
    setattr(torch_model,
            'onnx_mapping',
            {
                'inputs': onnx_model.input_names(),
                'outputs': onnx_model.output_names(),
            }
            )
    return torch_model
