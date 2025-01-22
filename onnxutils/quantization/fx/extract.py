from typing import Tuple

import torch


def extract_graph(graph: torch.fx.Graph, nodes: Tuple[torch.fx.Node]):
    new_graph = torch.fx.Graph()
    val_map = {}
    new_graph.graph_copy(graph, val_map)
    new_graph.output(tuple(val_map[node] for node in nodes))
    return new_graph


def extract(gm: torch.fx.GraphModule, nodes: Tuple[torch.fx.Node]):
    new_graph = extract_graph(gm.graph,  nodes)
    return torch.fx.GraphModule(gm, new_graph)
