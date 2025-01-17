import torch

from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from .pass import BasePass


class FakeQuantizeFinalizer(BasePass):
    @staticmethod
    def apply(graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            fq_mod = FakeQuantizeFinalizer.get_module_of_node(
                graph_module,
                node,
                (ObserverBase, FakeQuantizeBase)
            )
            if fq_mod is None:
                continue

            device = next(iter(graph_module.parameters())).device
            parent_name, _ = FakeQuantizeFinalizer.partition_module_name(
                node.target)
            parent_mod = graph_module.get_submodule(parent_name)

            qscheme = fq_mod.qscheme
            quant_min = fq_mod.quant_min
            quant_max = fq_mod.quant_max
            scale, zero_point = fq_mod.calculate_qparams()

            qparams = {
                'qscheme': qscheme,
                'quant_min': quant_min,
                'quant_max': quant_max,
                'scale': scale.to(torch.float),
                'zero_point': zero_point.to(torch.int32),
            }

            if qscheme in (
                    torch.per_tensor_affine,
                    torch.per_tensor_symmetric):
                op_func = torch.fake_quantize_per_tensor_affine
            elif qscheme in (
                    torch.per_channel_affine,
                    torch.per_channel_symmetric):
                op_func = torch.fake_quantize_per_channel_affine
                qparams['ch_axis'] = fq_mod.ch_axis
            else:
                raise NotImplementedError

            with graph_module.graph.inserting_before(node):
                op_args = [node.args[0]]

                for name in ['scale', 'zero_point',
                             'ch_axis',
                             'quant_min', 'quant_max']:
                    value = qparams.get(name, None)
                    if value is None:
                        continue

                    if name in ('scale', 'zero_point'):
                        new_value = (
                            value.detach().clone()
                            if isinstance(value, torch.Tensor)
                            else torch.tensor(value, device=device)
                        )
                        attr_name = FakeQuantizeFinalizer.get_new_attr_name_with_prefix(
                            parent_mod,
                            name)
                        parent_mod.register_buffer(attr_name, new_value)
                        attr_node = graph_module.graph.create_node(
                            'get_attr',
                            f"{parent_name}.{attr_name}"
                            if parent_name else attr_name
                        )
                        op_args.append(attr_node)
                    else:
                        op_args.append(value)

                op_node = graph_module.graph.create_node(
                    'call_function',
                    op_func,
                )
                node.replace_all_uses_with(op_node)
                graph_module.graph.erase_node(node)
                op_node.args = tuple(op_args)
        return torch.fx.GraphModule(graph_module, graph_module.graph)
