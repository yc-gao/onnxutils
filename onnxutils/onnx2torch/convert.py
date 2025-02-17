from operator import getitem
import warnings

import torch

from onnxutils.onnx import OnnxModel

from .node_converter.registry import find_converter


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

    root_module = torch.nn.Module()
    root_graph = torch.fx.Graph()

    nodes_mapping = {}

    # create input nodes
    for idx, name in enumerate(onnx_model.input_names()):
        if keep_input_names:
            if not name.isidentifier():
                raise ValueError(
                    f"Input name '{name}' cannot be used as name of placeholder in fx.GraphModule.")
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

        args = []
        for value_name in onnx_mapping.get('inputs', []):
            if onnx_model.get_input_by_name(value_name):
                args.append(nodes_mapping[value_name])
            elif initializer_value := onnx_model.get_initializer_by_name(value_name):
                if value_name not in nodes_mapping:
                    initializer_value = initializer_value.to_torch()
                    initializer_value.onnx_mapping = {
                        'name': value_name,
                    }
                    buffer_count = sum(1 for _ in root_module.buffers())
                    buffer_name = f'onnx_initializer_{buffer_count}'
                    root_module.register_buffer(buffer_name, initializer_value)
                    nodes_mapping[value_name] = root_graph.get_attr(
                        buffer_name)
                args.append(nodes_mapping[value_name])
            elif onnx_input_node := onnx_model.get_node_by_output(value_name):
                torch_input_node = nodes_mapping[onnx_input_node.name()]
                if len(onnx_input_node.outputs()) > 1:
                    index = onnx_input_node.outputs().index(value_name)
                    maybe_torch_input_node = nodes_mapping.get(
                        (torch_input_node, index),
                        None)
                    if maybe_torch_input_node is None:
                        maybe_torch_input_node = root_graph.call_function(
                            getitem, args=(torch_input_node, index))
                        nodes_mapping[(torch_input_node, index)
                                      ] = maybe_torch_input_node
                    torch_input_node = maybe_torch_input_node
                args.append(torch_input_node)
            else:
                warnings.warn(
                    f'Got unexpected input value name ({value_name})')

        nodes_mapping[onnx_node.name()] = root_graph.call_module(
            module_name=normalize_module_name(onnx_node.name()),
            args=tuple(args))

    args = []
    for output_name in onnx_model.output_names():
        onnx_output_node = onnx_model.get_node_by_output(output_name)
        assert onnx_output_node is not None
        torch_output_node = nodes_mapping[onnx_output_node.name()]
        if len(onnx_output_node.outputs()) > 1:
            index = onnx_output_node.outputs().index(output_name)
            torch_output_node = root_graph.call_function(
                getitem, args=(torch_output_node, index))
        args.append(torch_output_node)

    if len(args) > 1:
        root_graph.output(tuple(args))
    else:
        root_graph.output(args[0])

    root_graph.lint()

    torch_model = torch.fx.GraphModule(root=root_module, graph=root_graph)
    setattr(torch_model,
            'onnx_mapping',
            {
                'inputs': onnx_model.input_names(),
                'outputs': onnx_model.output_names(),
            }
            )
    return torch_model
