#!/bin/bash

export CUDA_VISIBLE_DEVICES=6,7
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
LOG_PREFIX=llama
TENSOR_PARALLEL_SIZE=2

input_lens=(512 1024 2048)
nano_batch_splits=(true)
use_nsys=false

for input_len in ${input_lens[@]}; do
  for nano_batch_split in ${nano_batch_splits[@]}; do
    if [ $use_nsys == true ]; then
      VLLM_ATTENTION_BACKEND=FLASH_ATTN \
      nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o tmp.nsys-rep -f true \
      vllm bench throughput \
        --model $MODEL_NAME --n 1 --num-prompts 1024 --input-len $input_len --output-len 128 \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --gpu_memory_utilization 0.9 --max-num-seqs 4096 \
        --compilation-config \{\"cudagraph_mode\":\ \"NONE\",\ \"enable_nano_batch_split\":\ $nano_batch_split,\ \"min_nano_split_tokens\":\ 2048\} \
        >${LOG_PREFIX}_tp${TENSOR_PARALLEL_SIZE}_${input_len}_${nano_batch_split}_nsys.log 2>&1
    else
      VLLM_ATTENTION_BACKEND=FLASH_ATTN \
      vllm bench throughput \
        --model $MODEL_NAME --n 1 --num-prompts 1024 --input-len $input_len --output-len 128 \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --gpu_memory_utilization 0.9 --max-num-seqs 4096 \
        --compilation-config \{\"cudagraph_mode\":\ \"NONE\",\ \"enable_nano_batch_split\":\ $nano_batch_split,\ \"min_nano_split_tokens\":\ 2048\} \
        >${LOG_PREFIX}_tp${TENSOR_PARALLEL_SIZE}_${input_len}_${nano_batch_split}.log 2>&1
    fi
  done
done
