#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
MODEL_NAME=Qwen/Qwen2-57B-A14B-Instruct
LOG_PREFIX=qwen
DATA_PARALLEL_SIZE=4

input_lens=(512 1024 2048)
nano_batch_splits=(true false)
use_nsys=false

for input_len in ${input_lens[@]}; do
  for nano_batch_split in ${nano_batch_splits[@]}; do
    if [ $use_nsys == true ]; then
      VLLM_ATTENTION_BACKEND=FLASH_ATTN \
      nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o tmp.nsys-rep -f true \
      python benchmarks/benchmark_data_parallel.py \
        --model $MODEL_NAME --num-prompts 1024 --input-len $input_len --output-len 128 \
        --dp-size $DATA_PARALLEL_SIZE \
        --gpu_memory_utilization 0.9 --max-num-seqs 4096 \
        --compilation-config \{\"cudagraph_mode\":\ \"NONE\",\ \"enable_nano_batch_split\":\ $nano_batch_split,\ \"min_nano_split_tokens\":\ 2048\} \
        >${LOG_PREFIX}_dp${DATA_PARALLEL_SIZE}_${input_len}_${nano_batch_split}_nsys.log 2>&1
    else
      VLLM_ATTENTION_BACKEND=FLASH_ATTN \
      python benchmarks/benchmark_data_parallel.py \
        --model $MODEL_NAME --num-prompts 4096 --input-len $input_len --output-len 128 \
        --dp-size $DATA_PARALLEL_SIZE \
        --gpu_memory_utilization 0.9 --max-num-seqs 4096 \
        --compilation-config \{\"cudagraph_mode\":\ \"NONE\",\ \"enable_nano_batch_split\":\ $nano_batch_split,\ \"min_nano_split_tokens\":\ 2048\} \
        >${LOG_PREFIX}_dp${DATA_PARALLEL_SIZE}_${input_len}_${nano_batch_split}.log 2>&1
    fi
  done
done
