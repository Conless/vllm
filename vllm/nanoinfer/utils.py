# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


def tag_graph(gm: torch.fx.GraphModule, op_tags: dict[str, str]):
    submodules = [(name, module) for (name, module) in gm.named_modules()
                  if hasattr(module, "graph")]
    for _, module in submodules:
        for node in module.graph.nodes:
            if (node.op == "call_function"
                    and (tag := op_tags.get(str(node.target))) is not None):
                assert (getattr(module, "tag", None) is None or module.tag
                        == tag), f"tag mismatch: {module.tag} != {tag}"
                module.tag = tag
