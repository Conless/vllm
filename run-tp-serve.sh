#!/bin/bash

MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct
LOG_PREFIX=llama
TENSOR_PARALLEL_SIZE=8
DEVICES=0,1,2,3,4,5,6,7
nano_batch_split=true
use_nsys=false
rps=(10 20 30 40 50 60)

if [ $use_nsys == true ]; then
  VLLM_ATTENTION_BACKEND=FLASH_ATTN CUDA_VISIBLE_DEVICES=${DEVICES} \
      nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o tmp.nsys-rep -f true \
  vllm serve ${MODEL_NAME} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_utilization 0.9 --max-num-seqs 4096 --no-enable-prefix-caching \
    --compilation-config \{\"cudagraph_mode\":\ \"NONE\",\ \"enable_nano_batch_split\":\ $nano_batch_split,\ \"min_nano_split_tokens\":\ 2048\} \
  >${LOG_PREFIX}_serve_tp${TENSOR_PARALLEL_SIZE}_${nano_batch_split}_nsys.log 2>&1 &
else
  VLLM_ATTENTION_BACKEND=FLASH_ATTN CUDA_VISIBLE_DEVICES=${DEVICES} \
  vllm serve ${MODEL_NAME} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --gpu_memory_utilization 0.9 --max-num-seqs 4096 --no-enable-prefix-caching \
  --compilation-config \{\"cudagraph_mode\":\ \"NONE\",\ \"enable_nano_batch_split\":\ $nano_batch_split,\ \"min_nano_split_tokens\":\ 2048\} \
  >${LOG_PREFIX}_serve_tp${TENSOR_PARALLEL_SIZE}_${nano_batch_split}.log 2>&1 &
fi

for rps in ${rps[@]}; do
  vllm bench serve \
    --model ${MODEL_NAME} \
    --dataset-name sharegpt \
    --dataset-path /root/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 4096 \
    --request-rate $rps \
  >${LOG_PREFIX}_serve_tp${TENSOR_PARALLEL_SIZE}_${nano_batch_split}_rps${rps}.log 2>&1
done
