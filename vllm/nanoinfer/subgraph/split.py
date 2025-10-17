# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.fx as fx


@dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


def split_graph(graph: fx.GraphModule, ops: List[str]) -> Tuple[fx.GraphModule, List[SplitItem]]:
    """
    Split a traced fullgraph into subgraphs by op boundaries (if ops provided),
    producing a stitched GraphModule with submodules submod_0, submod_1, ...
    """
    node_to_subgraph_id: dict[fx.Node, int] = {}
    subgraph_id = 0
    split_op_graphs: List[int] = []

    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == 'call_function' and str(node.target) in ops:
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    split_gm = torch.fx.passes.split_module.split_module(  # type: ignore[attr-defined]
        graph,
        None,
        lambda n: node_to_subgraph_id.get(n, 0),
        keep_original_order=True,
    )

    items: List[SplitItem] = []
    names = [name for (name, module) in split_gm.named_modules()]
    for name in names:
        if "." in name or name == "":
            continue
        module = getattr(split_gm, name)
        gid = int(name.replace("submod_", "")) if name.startswith("submod_") else 0
        items.append(
            SplitItem(
                submod_name=name,
                graph_id=gid,
                is_splitting_graph=(gid in split_op_graphs),
                graph=module,
            ))
    items.sort(key=lambda x: x.graph_id)
    return split_gm, items


