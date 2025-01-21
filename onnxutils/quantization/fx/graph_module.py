from typing import Any, Union, Tuple

import torch


class QuantizedGraphModule(torch.fx.GraphModule):
    def __init__(self,
                 root: Union[torch.nn.Module, dict[str, Any]],
                 graph: torch.fx.Graph,
                 preserved_attr_names: set[str]
                 ) -> None:
        self.preserved_attr_names = preserved_attr_names
        preserved_attrs = {
            attr: getattr(root, attr)
            for attr in self.preserved_attr_names
            if hasattr(root, attr)
        }
        super().__init__(root, graph)

        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    def extract(self, nodes: Tuple[torch.fx.Node]):
        new_graph = torch.fx.Graph()
        val_map = {}
        new_graph.graph_copy(self.graph, val_map)
        new_graph.output(tuple(val_map[node] for node in nodes))
        return QuantizedGraphModule(self, new_graph, self.preserved_attr_names)
