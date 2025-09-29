# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import dataclasses
import os
from time import sleep
from typing import Union

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import RandomDataset, SampleRequest
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs.data import TextPrompt, TokensPrompt
from vllm.utils import FlexibleArgumentParser, get_open_port


def create_argument_parser():
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Input prompt length for each request",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the "
        "output length from the dataset.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )

    parser = EngineArgs.add_cli_args(parser)

    return parser


def get_requests(
    args: argparse.Namespace, tokenizer: AutoTokenizer
) -> list[SampleRequest]:
    sample_kwargs = {
        "tokenizer": tokenizer,
        "num_requests": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "prefix_len": 0,
        "random_range_ratio": None,
    }
    # Remove None values
    sample_kwargs = {k: v for k, v in sample_kwargs.items() if v is not None}
    return RandomDataset(random_seed=0).sample(**sample_kwargs)


def prepare_inputs(
    args: argparse.Namespace,
) -> tuple[list[Union[TextPrompt, TokensPrompt]], list[SamplingParams]]:
    # Sample prompts.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    requests = get_requests(args, tokenizer)
    prompts: list[Union[TextPrompt, TokensPrompt]] = []
    sampling_params: list[SamplingParams] = []
    for request in requests:
        prompts.append(
            TokensPrompt(
                prompt_token_ids=request.prompt["prompt_token_ids"],  # type: ignore
                multi_modal_data=request.multi_modal_data,  # type: ignore
            )
            if "prompt_token_ids" in request.prompt
            else TextPrompt(
                prompt=request.prompt,
                multi_modal_data=request.multi_modal_data,  # type: ignore
            )
        )
        sampling_params.append(
            SamplingParams(
                n=1,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                detokenize=False,
            )
        )
    return prompts, sampling_params


def main(
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    engine_args: EngineArgs,
    prompts: list[Union[TextPrompt, TokensPrompt]],
    sampling_params: list[SamplingParams],
):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # CUDA_VISIBLE_DEVICES for each DP rank is set automatically inside the
    # engine processes.

    print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

    # Create an LLM.
    llm = LLM(**dataclasses.asdict(engine_args))
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for i, output in enumerate(outputs):
        if i >= 5:
            # print only 5 outputs
            break
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(
            f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
            f"Generated text: {generated_text!r}"
        )

    # Give engines time to pause their processing loops before exiting.
    sleep(1)


if __name__ == "__main__":
    args = create_argument_parser().parse_args()
    assert args is not None
    if args.tokenizer is None:
        args.tokenizer = args.model
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.disable_log_stats = True

    dp_size = 2
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()

    prompts, sampling_params = prepare_inputs(args)
    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)

    from multiprocessing import Process

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(range(0, dp_size)):
        proc = Process(
            target=main,
            args=(
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                engine_args,
                prompts[start(global_dp_rank) : start(global_dp_rank + 1)],
                sampling_params[start(global_dp_rank) : start(global_dp_rank + 1)],
            ),
        )
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join(timeout=1200)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop within 2 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
