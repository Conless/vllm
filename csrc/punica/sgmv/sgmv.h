#pragma once
#include <cuda_runtime.h>

#include <cstdint>

template <typename DType>
bool sgmv(DType* y, DType* x, DType** w, int32_t* s, void* tmp_d,
          int num_problems, int d_in, int d_out, int layer_idx,
          cudaStream_t stream);

template <typename YType, typename XType, typename WType>
bool sgmv_custom(YType* y, XType* x, WType** w, int32_t* s, void* tmp_d,
          int num_problems, int d_in, int d_out, int layer_idx,
          cudaStream_t stream);

size_t sgmv_tmp_size(int num_problems);