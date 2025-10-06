# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.logprobs import Logprob, PromptLogprobs, SampleLogprobs
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_ids_list_to_tokens)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

logger = init_logger(__name__)

NONES = itertools.repeat(None)


@dataclass
class LogprobsProcessor:

    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: Optional[AnyTokenizer]

    # Logprobs for this request
    logprobs: Optional[LogprobsLists]
    prompt_logprobs: Optional[LogprobsTensors]
    cumulative_logprob: Optional[float]
    num_logprobs: Optional[int]
    num_prompt_logprobs: Optional[int]

    @classmethod
    def from_new_request(
        cls,
        tokenizer: Optional[AnyTokenizer],
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        assert request.sampling_params is not None
        num_logprobs = request.sampling_params.logprobs
        num_prompt_logprobs = request.sampling_params.prompt_logprobs
        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.),
            logprobs=None,
            # NOTE: logprob of first prompt token is None.
            prompt_logprobs=None,
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
        )

    def pop_prompt_logprobs(self) -> Optional[LogprobsTensors]:
        """Pop and return all request prompt logprobs

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.
        """
        plp = self.prompt_logprobs
        if plp:
            self.prompt_logprobs = None
        return plp

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if self.logprobs is None:
            self.logprobs = output.new_logprobs
        elif output.new_logprobs is not None:
            self.logprobs.logprob_token_ids.extend(output.new_logprobs.logprob_token_ids)
            self.logprobs.logprobs.extend(output.new_logprobs.logprobs)
            self.logprobs.sampled_token_ranks.extend(output.new_logprobs.sampled_token_ranks)
        if self.prompt_logprobs is None:
            self.prompt_logprobs = output.new_prompt_logprobs_tensors
        elif output.new_prompt_logprobs_tensors is not None:
            self.prompt_logprobs = LogprobsTensors(
                logprob_token_ids=torch.cat(
                    [
                        self.prompt_logprobs.logprob_token_ids,
                        output.new_prompt_logprobs_tensors.logprob_token_ids,
                    ],
                    dim=0,
                ),
                logprobs=torch.cat(
                    [
                        self.prompt_logprobs.logprobs,
                        output.new_prompt_logprobs_tensors.logprobs,
                    ],
                    dim=0,
                ),
                selected_token_ranks=torch.cat(
                    [
                        self.prompt_logprobs.selected_token_ranks,
                        output.new_prompt_logprobs_tensors.selected_token_ranks,
                    ],
                    dim=0,
                ),
            )
