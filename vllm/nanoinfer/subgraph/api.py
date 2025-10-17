# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Any

import torch.fx as fx

from .split import split_graph
from .interpreter import SubgraphCompileInterpreter
from .config import SubgraphConfig


def compile_subgraphs(fullgraph: fx.GraphModule, cfg: SubgraphConfig, example_inputs: List[Any]) -> fx.GraphModule:
    assert cfg.splitting_ops is not None and len(cfg.splitting_ops) > 0, "splitting_ops must be provided to produce stitched subgraphs"
    stitched_gm, items = split_graph(fullgraph, cfg.splitting_ops)

    # 2) Select submodules: compile all non-splitting subgraphs
    targets = [it.submod_name for it in items if not it.is_splitting_graph]

    # 3) Install backends via interpreter, run with example inputs
    SubgraphCompileInterpreter(stitched_gm, targets, cfg).run(*example_inputs)

    return stitched_gm


