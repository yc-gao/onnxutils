import torch

from .quantizer import QuantizerBase


class NodeQuantizer(QuantizerBase):
    def quantize(
        self,
        graph_module: torch.fx.GraphModule,
        qconfigs: list[dict]
    ):
        qconfig_mapping: dict[str, dict] = {
            qconfig['name']: qconfig for qconfig in qconfigs
        }

        for node in graph_module.graph.nodes:
            qconfig = qconfig_mapping.get(node.name, None)
            if qconfig is None:
                continue

            act_qconfig = qconfig.get('activation', None)
            if act_qconfig is None:
                continue

            fq_mod = act_qconfig()
            fq_name = self.get_new_attr_name_with_prefix(
                graph_module, 'fq')
            graph_module.add_submodule(fq_name, fq_mod)

            with graph_module.graph.inserting_after(node):
                fq_node = graph_module.graph.create_node(
                    'call_module',
                    fq_name
                )
                node.replace_all_uses_with(fq_node)
                fq_node.args = (node,)

        return torch.fx.GraphModule(graph_module, graph_module.graph)
