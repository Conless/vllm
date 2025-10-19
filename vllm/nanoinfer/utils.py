# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any
import torch

from vllm.nanoinfer.config import NanoInferConfig
from vllm.nanoinfer.backend import SubgraphCompileInterpreter


def split_graph(
    graph: torch.fx.GraphModule, splitting_ops: list[str]
) -> torch.fx.GraphModule:
    """
    Split a traced fullgraph into subgraphs by op boundaries (if ops provided),
    producing a stitched GraphModule with submodules submod_0, submod_1, ...
    """
    node_to_subgraph_id: dict[torch.fx.Node, int] = {}
    subgraph_id = 0
    split_op_graphs: list[int] = []

    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == "call_function" and str(node.target) in splitting_ops:
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

    return split_gm


def tag_graph(gm: torch.fx.GraphModule, op_tags: dict[str, str]) -> None:
    submodules = [
        (name, module)
        for (name, module) in gm.named_modules()
        if hasattr(module, "graph")
    ]
    for name, module in submodules:
        if "." in name or name == "":
            continue
        module.tag = "" # type: ignore
        for node in module.graph.nodes:
            if (
                node.op == "call_function"
                and (tag := op_tags.get(str(node.target))) is not None
            ):
                assert (
                    module.tag == "" or module.tag == tag
                ), f"tag mismatch: {module.tag} != {tag}"
                module.tag = tag # type: ignore


def compile_subgraphs(
    fullgraph: torch.fx.GraphModule,
    config: NanoInferConfig,
    example_inputs: list[Any],
) -> torch.fx.GraphModule:
    stitched_gm = split_graph(fullgraph, config.splitting_ops)
    tag_graph(stitched_gm, config.special_ops)
    targets = [name for name, module in stitched_gm.named_modules() if hasattr(module, "tag") and module.tag != "memory"]
    print("targets: ", targets)
    SubgraphCompileInterpreter(stitched_gm, targets, config).run(
        *example_inputs
    )
    return stitched_gm


def pack_tokens(num_tokens: int, cudagraph_capture_sizes: list[int]) -> int:
    if num_tokens <= max(cudagraph_capture_sizes):
        return min(
            size for size in cudagraph_capture_sizes if size >= num_tokens
        )
    return num_tokens
