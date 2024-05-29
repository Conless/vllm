#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

template <typename T>
struct cutlass_dtype {
  using type = T;
};

template <>
struct cutlass_dtype<float> {
  using type = float;
};

template <>
struct cutlass_dtype<half> {
  using type = cutlass::half_t;
};

template <>
struct cutlass_dtype<nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename YType, typename XType, typename WType>
__global__ void precompute_sgmv_args(cutlass::gemm::GemmCoord *all_problems,
                                     YType **ptr_y, XType **ptr_x, WType **ptr_w,
                                     int64_t *ld_y, int64_t *ld_x,
                                     int64_t *ld_w, YType *y, XType *x, WType **w,
                                     int32_t *s, int d_in, int d_out,
                                     int layer_idx) {
  int i = blockIdx.x;
  int m = s[i + 1] - s[i], k = d_in, n = d_out;
  all_problems[i] = cutlass::gemm::GemmCoord(m, n, k);
  ptr_w[i] = w[i] + layer_idx * d_in * d_out;
  ptr_x[i] = x + s[i] * d_in;
  ptr_y[i] = y + s[i] * d_out;
  ld_x[i] = k;
  ld_w[i] = n;
  ld_y[i] = n;
}

size_t sgmv_tmp_size(int num_problems) {
  constexpr auto sz = sizeof(void *) * 3 + sizeof(int64_t) * 3 +
                      sizeof(cutlass::gemm::GemmCoord);
  return sz * num_problems;
}

template <typename T>
inline T *alloc_from_buf(void **buf, int n) {
  auto *p = (T *)*buf;
  *buf = (void *)(p + n);
  return p;
}

template <typename DType>
bool sgmv(DType *y, DType *x, DType **w, int32_t *s, void *tmp_d,
          int num_problems, int d_in, int d_out, int layer_idx,
          cudaStream_t stream) {
  using cutlass_t = typename cutlass_dtype<DType>::type;

  auto ptr_Y = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);
  auto ptr_X = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);
  auto ptr_W = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);
  auto ld_Y = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_X = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_W = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto all_problems =
      alloc_from_buf<cutlass::gemm::GemmCoord>(&tmp_d, num_problems);

  precompute_sgmv_args<<<num_problems, 1, 0, stream>>>(
      all_problems, ptr_Y, ptr_X, ptr_W, ld_Y, ld_X, ld_W, (cutlass_t *)y,
      (cutlass_t *)x, (cutlass_t **)w, s, d_in, d_out, layer_idx);

  using cutlass::epilogue::thread::LinearCombination;
  using cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle;
  if (d_in < d_out) {
    // Expand
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        cutlass_t,                                      // Element A
        cutlass::layout::RowMajor,                      // Layout A
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity A
        cutlass_t,                                      // Element B
        cutlass::layout::RowMajor,                      // Layout B
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity B
        cutlass_t,                                      // Element C&D
        cutlass::layout::RowMajor,                      // Layout C&D
        float,                                          // Element Accumulator
        cutlass::arch::OpClassTensorOp,                 // Operator Class Tag
        cutlass::arch::Sm80,                            // Architecture
        cutlass::gemm::GemmShape<32, 128, 16>,          // Thread Block Shape
        cutlass::gemm::GemmShape<32, 64, 16>,           // Warp Shape
        cutlass::gemm::GemmShape<16, 8, 8>,             // Instruction Shape
        LinearCombination<cutlass_t, 8, float, float>,  // Epilogue
        GemmIdentityThreadblockSwizzle<1>,              // Swizzling Operator
        2                                               // Stages
        >::GemmKernel;

    using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
    typename EpilogueOutputOp::Params epilogue_op(1.0, 1.0);

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
    typename GemmGrouped::Arguments args(all_problems, num_problems, 512,
                                         epilogue_op, ptr_X, ptr_W, ptr_Y,
                                         ptr_Y, ld_X, ld_W, ld_Y, ld_Y);

    GemmGrouped gemm;
    auto status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "sgmv_cutlass gemm.initialize failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "sgmv_cutlass gemm.run failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
  } else {
    // Shrink
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        cutlass_t,                                      // Element A
        cutlass::layout::RowMajor,                      // Layout A
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity A
        cutlass_t,                                      // Element B
        cutlass::layout::RowMajor,                      // Layout B
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity B
        cutlass_t,                                      // Element C&D
        cutlass::layout::RowMajor,                      // Layout C&D
        float,                                          // Element Accumulator
        cutlass::arch::OpClassTensorOp,                 // Operator Class Tag
        cutlass::arch::Sm80,                            // Architecture
        cutlass::gemm::GemmShape<16, 64, 64>,           // Thread Block Shape
        cutlass::gemm::GemmShape<16, 16, 64>,           // Warp Shape
        cutlass::gemm::GemmShape<16, 8, 16>,            // Instruction Shape
        LinearCombination<cutlass_t, 4, float, float>,  // Epilogue
        GemmIdentityThreadblockSwizzle<2>,              // Swizzling Operator
        2                                               // Stages
        >::GemmKernel;

    using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
    typename EpilogueOutputOp::Params epilogue_op(1.0, 1.0);

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
    typename GemmGrouped::Arguments args(all_problems, num_problems, 512,
                                         epilogue_op, ptr_X, ptr_W, ptr_Y,
                                         ptr_Y, ld_X, ld_W, ld_Y, ld_Y);

    GemmGrouped gemm;
    auto status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "sgmv_cutlass gemm.initialize failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "sgmv_cutlass gemm.run failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
  }
  return true;
}

// In Punica, YType, XType and WType are required to be all the same.
// However, the precision would be sacrificed in LoRA computation,
// where we wanna use float32 for intermediate results.
// So we make a modification here to allow different precisions.
template <typename YType, typename XType, typename WType>
bool sgmv_custom(YType *y, XType *x, WType **w, int32_t *s, void *tmp_d,
          int num_problems, int d_in, int d_out, int layer_idx,
          cudaStream_t stream) {
  using cutlass_y_t = typename cutlass_dtype<YType>::type;
  using cutlass_x_t = typename cutlass_dtype<XType>::type;
  using cutlass_w_t = typename cutlass_dtype<WType>::type;

  auto ptr_Y = alloc_from_buf<cutlass_y_t *>(&tmp_d, num_problems);
  auto ptr_X = alloc_from_buf<cutlass_x_t *>(&tmp_d, num_problems);
  auto ptr_W = alloc_from_buf<cutlass_w_t *>(&tmp_d, num_problems);
  auto ld_Y = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_X = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_W = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto all_problems =
      alloc_from_buf<cutlass::gemm::GemmCoord>(&tmp_d, num_problems);

  precompute_sgmv_args<<<num_problems, 1, 0, stream>>>(
      all_problems, ptr_Y, ptr_X, ptr_W, ld_Y, ld_X, ld_W, (cutlass_y_t *)y,
      (cutlass_x_t *)x, (cutlass_w_t **)w, s, d_in, d_out, layer_idx);

  using cutlass::epilogue::thread::LinearCombination;
  using cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle;
  if (d_in < d_out) {
    // Expand
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        cutlass_x_t,                                    // Element A
        cutlass::layout::RowMajor,                      // Layout A
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity A
        cutlass_w_t,                                    // Element B
        cutlass::layout::RowMajor,                      // Layout B
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity B
        cutlass_y_t,                                    // Element C&D
        cutlass::layout::RowMajor,                      // Layout C&D
        float,                                          // Element Accumulator
        cutlass::arch::OpClassTensorOp,                 // Operator Class Tag
        cutlass::arch::Sm80,                            // Architecture
        cutlass::gemm::GemmShape<32, 128, 16>,          // Thread Block Shape
        cutlass::gemm::GemmShape<32, 64, 16>,           // Warp Shape
        cutlass::gemm::GemmShape<16, 8, 8>,             // Instruction Shape
        LinearCombination<cutlass_y_t, 8, float, float>,// Epilogue
        GemmIdentityThreadblockSwizzle<1>,              // Swizzling Operator
        2                                               // Stages
        >::GemmKernel;

    using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
    typename EpilogueOutputOp::Params epilogue_op(1.0, 1.0);

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
    typename GemmGrouped::Arguments args(all_problems, num_problems, 512,
                                         epilogue_op, ptr_X, ptr_W, ptr_Y,
                                         ptr_Y, ld_X, ld_W, ld_Y, ld_Y);

    GemmGrouped gemm;
    auto status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "sgmv_cutlass gemm.initialize failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "sgmv_cutlass gemm.run failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
  } else {
    // Shrink
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        cutlass_x_t,                                    // Element A
        cutlass::layout::RowMajor,                      // Layout A
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity A
        cutlass_w_t,                                    // Element B
        cutlass::layout::RowMajor,                      // Layout B
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity B
        cutlass_y_t,                                      // Element C&D
        cutlass::layout::RowMajor,                      // Layout C&D
        float,                                          // Element Accumulator
        cutlass::arch::OpClassTensorOp,                 // Operator Class Tag
        cutlass::arch::Sm80,                            // Architecture
        cutlass::gemm::GemmShape<16, 64, 64>,           // Thread Block Shape
        cutlass::gemm::GemmShape<16, 16, 64>,           // Warp Shape
        cutlass::gemm::GemmShape<16, 8, 16>,            // Instruction Shape
        LinearCombination<cutlass_y_t, 4, float, float>,  // Epilogue
        GemmIdentityThreadblockSwizzle<2>,              // Swizzling Operator
        2                                               // Stages
        >::GemmKernel;

    using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
    typename EpilogueOutputOp::Params epilogue_op(1.0, 1.0);

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
    typename GemmGrouped::Arguments args(all_problems, num_problems, 512,
                                         epilogue_op, ptr_X, ptr_W, ptr_Y,
                                         ptr_Y, ld_X, ld_W, ld_Y, ld_Y);

    GemmGrouped gemm;
    auto status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "sgmv_cutlass gemm.initialize failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "sgmv_cutlass gemm.run failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
  }
  return true;
}
