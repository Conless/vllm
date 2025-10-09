import asyncio
from collections import deque
import contextlib
from typing import Any, Callable, Optional

import torch

from vllm.nanoinfer.interface import OperatorHandle, SplitConfig


class NanoInferEngine:
    """Engine that executes FX graph with software-defined operator scheduling.

    The engine maintains two concurrent tasks:
    - Producer: Pushes ready module operators to op_queue
    - Consumer: Processes execution requests from execute_queue

    Non-module operations (call_function, call_method, output) are executed
    automatically in the background without scheduler involvement.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        split_config: SplitConfig,
        hook: Optional[
            Callable[
                [tuple[OperatorHandle]], contextlib.AbstractContextManager[None]
            ]
        ] = None,
    ):
        self.graph_module = graph_module
        self.split_config = split_config
        self.hook = hook
        self.num_nano_batches = split_config.num_nano_batches

        self.env: dict[int, dict[torch.fx.Node, Any]] = {
            i: {} for i in range(self.num_nano_batches)
        }

        self.results: dict[int, Any] = {}

        self.node_queue: dict[int, deque[torch.fx.Node]] = {
            i: deque(self.graph_module.graph.nodes)
            for i in range(self.num_nano_batches)
        }

        self.module_name_to_node: dict[str, torch.fx.Node] = {}
        for node in self.graph_module.graph.nodes:
            if node.op == "call_module":
                assert isinstance(node.target, str)
                self.module_name_to_node[node.target] = node

        self.last_event: dict[int, Optional[torch.cuda.Event]] = {
            i: None for i in range(self.num_nano_batches)
        }

    async def execute(
        self,
        args: tuple,
        kwargs: dict,
        op_queue: dict[int, asyncio.Queue[OperatorHandle | None]],
        execute_queue: asyncio.Queue[
            tuple[
                tuple[OperatorHandle],
                Callable | None,
                asyncio.Event,
            ]
        ],
    ) -> dict[int, Any]:
        """Execute the model with simplified single-loop pattern.

        Args:
            args: Input arguments
            kwargs: Input keyword arguments
            op_queue: Queues to push ready operators (one per nano-batch)
            execute_queue: Queue to receive execution requests from scheduler

        Returns:
            Dictionary mapping nano-batch index to results
        """
        # Pre-loop: Execute all nodes before first module in each batch
        for batch_idx in range(self.num_nano_batches):
            while self.node_queue[batch_idx]:
                node = self.node_queue[batch_idx][0]
                if node.op == "call_module" or node.op == "output":
                    break
                self._execute_non_module(node, batch_idx, args)
                self.node_queue[batch_idx].popleft()

        # Track pushed operators for each batch
        pushed_operators: dict[int, list[OperatorHandle]] = {
            i: [] for i in range(self.num_nano_batches)
        }

        # Main loop
        while any(self.node_queue.values()):
            for batch_idx in range(self.num_nano_batches):
                while self.node_queue[batch_idx]:
                    if op_queue[batch_idx].full():
                        break
                    node = self.node_queue[batch_idx][0]
                    if node.op == "call_module":
                        assert isinstance(node.target, str)
                        module = getattr(self.graph_module, node.target)
                        op_handle = OperatorHandle(
                            module_name=node.target,
                            nano_batch_idx=batch_idx,
                            debug_info={"tag": getattr(module, "tag", "")},
                        )
                        await op_queue[batch_idx].put(op_handle)
                        pushed_operators[batch_idx].append(op_handle)
                        self.node_queue[batch_idx].popleft()
                    else:
                        break

            item = await execute_queue.get()
            operators, func, done_event = item
            node_args = []
            node_kwargs = []
            for op in operators:
                batch_idx = op.nano_batch_idx
                node = self.module_name_to_node[op.module_name]
                if self.last_event[batch_idx] is not None:
                    last_event = self.last_event[batch_idx]
                    assert last_event is not None
                    last_event.wait()
                self.last_event[batch_idx] = torch.cuda.Event()
                node_args.append(
                    [
                        self.env[batch_idx][arg]
                        for arg in node.args
                        if isinstance(arg, torch.fx.Node)
                    ]
                )
                node_kwargs.append(
                    {
                        k: self.env[batch_idx][v]
                        if isinstance(v, torch.fx.Node)
                        else v
                        for k, v in node.kwargs.items()
                    }
                )
            node_args = tuple(node_args)
            node_kwargs = tuple(node_kwargs)
            results = []
            if func is not None:
                with (
                    self.hook(operators)
                    if self.hook is not None
                    else contextlib.nullcontext()
                ):
                    results = func(node_args, node_kwargs)
            else:
                for idx, op in enumerate(operators):
                    with (
                        self.hook((op,))
                        if self.hook is not None
                        else contextlib.nullcontext()
                    ), torch.cuda.nvtx.range(f"op_{op.module_name}_{op.nano_batch_idx}"):
                        module = getattr(self.graph_module, op.module_name)
                        # print(f"Executing module: {op.module_name}")
                        results.append(
                            module(*node_args[idx], **node_kwargs[idx])
                        )

            for op, result in zip(operators, results):
                batch_idx = op.nano_batch_idx
                node = self.module_name_to_node[op.module_name]
                last_event = self.last_event[batch_idx]
                assert last_event is not None
                last_event.record()
                self.env[batch_idx][node] = result
                assert op == pushed_operators[batch_idx][0]
                pushed_operators[batch_idx].pop(0)

            for batch_idx in range(self.num_nano_batches):
                if pushed_operators[batch_idx]:
                    continue
                while self.node_queue[batch_idx]:
                    node = self.node_queue[batch_idx][0]
                    if node.op == "call_module":
                        break
                    elif node.op == "output":
                        if isinstance(node.args[0], torch.fx.Node):
                            self.results[batch_idx] = self.env[batch_idx][
                                node.args[0]
                            ]
                        elif isinstance(node.args[0], (tuple, list)):
                            self.results[batch_idx] = tuple(
                                self.env[batch_idx][arg]
                                if isinstance(arg, torch.fx.Node)
                                else arg
                                for arg in node.args[0]
                            )
                        else:
                            self.results[batch_idx] = node.args[0]
                        self.node_queue[batch_idx].popleft()
                        await op_queue[batch_idx].put(None)
                        break
                    else:
                        self._execute_non_module(node, batch_idx, args)
                        self.node_queue[batch_idx].popleft()
            done_event.set()

        return self.results

    def _execute_non_module(
        self, node: torch.fx.Node, batch_idx: int, args: tuple
    ) -> None:
        """Execute non-module operations (placeholder, get_attr, call_function,
        call_method)."""
        if node.op == "placeholder":
            placeholder_nodes = [
                n
                for n in self.graph_module.graph.nodes
                if n.op == "placeholder"
            ]
            placeholder_idx = placeholder_nodes.index(node)

            example_value = node.meta.get("example_value", None)

            token_start = self.split_config.split_indices[batch_idx]
            token_end = self.split_config.split_indices[batch_idx + 1]
            num_tokens_in_batch = self.split_config.num_tokens[batch_idx]
            total_num_tokens = sum(self.split_config.num_tokens)

            if isinstance(example_value, torch.Tensor):
                if isinstance(example_value.shape[0], torch.SymInt):
                    self.env[batch_idx][node] = args[placeholder_idx][
                        token_start:token_end
                    ]
                else:
                    self.env[batch_idx][node] = args[placeholder_idx]
            elif isinstance(example_value, torch.SymInt):
                assert args[placeholder_idx] == total_num_tokens
                self.env[batch_idx][node] = num_tokens_in_batch
            else:
                raise ValueError(
                    f"Invalid example value type: {type(example_value)}"
                )
        elif node.op == "call_function":
            target = node.target
            assert callable(target)
            node_args = [
                self.env[batch_idx][arg]
                if isinstance(arg, torch.fx.Node)
                else arg
                for arg in node.args
            ]
            node_kwargs = {
                k: self.env[batch_idx][v] if isinstance(v, torch.fx.Node) else v
                for k, v in node.kwargs.items()
            }
            self.env[batch_idx][node] = target(*node_args, **node_kwargs)
        elif node.op == "call_method":
            assert isinstance(node.target, str)
            self_obj = self.env[batch_idx][node.args[0]]
            target = getattr(self_obj, node.target)
            assert callable(target)
            method_args = [
                self.env[batch_idx][arg]
                if isinstance(arg, torch.fx.Node)
                else arg
                for arg in node.args[1:]
            ]
            node_kwargs = {
                k: self.env[batch_idx][v] if isinstance(v, torch.fx.Node) else v
                for k, v in node.kwargs.items()
            }
            self.env[batch_idx][node] = target(*method_args, **node_kwargs)
        elif node.op == "get_attr":
            target = node.target
            assert isinstance(target, str)
            self.env[batch_idx][node] = getattr(self.graph_module, target)
        else:
            raise ValueError(f"Invalid node operation: {node.op}")
