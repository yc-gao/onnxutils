import torch

from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.fake_quantize import FakeQuantizeBase


def get_module(gm: torch.fx.GraphModule, node: torch.fx.Node):
    if node.op != 'call_module':
        return None
    return gm.get_submodule(node.target)


def get_new_attr_name_with_prefix(
        module: torch.nn.Module,
        prefix: str,
        idx: int = 0):
    prefix = prefix.replace(".", "_")

    while True:
        attr_name = f"{prefix}{idx}"
        if not hasattr(module, attr_name):
            break
        idx += 1
    return attr_name


def partition_module_name(target: str):
    *p, r = target.rsplit('.', 1)
    return '.'.join(p), r


class FakeQuantizeFinalizer:
    @staticmethod
    def apply(gm: torch.fx.GraphModule):
        for node in gm.graph.nodes:
            maybe_fq_mod = get_module(gm, node)
            if maybe_fq_mod is None:
                continue
            if not isinstance(maybe_fq_mod, (ObserverBase, FakeQuantizeBase)):
                continue

            parent_name, _ = partition_module_name(node.target)
            parent_mod = gm.get_submodule(parent_name)
            device = next(iter(gm.parameters())).device

            qscheme = maybe_fq_mod.qscheme
            quant_min = maybe_fq_mod.quant_min
            quant_max = maybe_fq_mod.quant_max
            scale, zero_point = maybe_fq_mod.calculate_qparams()

            qparams = {
                'qscheme': qscheme,
                'quant_min': quant_min,
                'quant_max': quant_max,
                'scale': scale.to(torch.float32),
                'zero_point': zero_point.to(torch.int64),
            }

            if qscheme in (
                    torch.per_tensor_affine,
                    torch.per_tensor_symmetric):
                op_func = torch.fake_quantize_per_tensor_affine
            elif qscheme in (
                    torch.per_channel_affine,
                    torch.per_channel_symmetric):
                op_func = torch.fake_quantize_per_channel_affine
                qparams['ch_axis'] = maybe_fq_mod.ch_axis
            else:
                raise NotImplementedError

            with gm.graph.inserting_before(node):
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
                        attr_name = get_new_attr_name_with_prefix(
                            parent_mod,
                            name)
                        parent_mod.register_buffer(attr_name, new_value)
                        attr_node = gm.graph.create_node(
                            'get_attr',
                            f"{parent_name}.{attr_name}"
                            if parent_name else attr_name
                        )
                        op_args.append(attr_node)
                    else:
                        op_args.append(value)

                op_node = gm.graph.create_node(
                    'call_function',
                    op_func,
                )
                node.replace_all_uses_with(op_node)
                gm.graph.erase_node(node)
                op_node.args = tuple(op_args)
        return torch.fx.GraphModule(gm, gm.graph)
