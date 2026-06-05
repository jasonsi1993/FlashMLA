#pragma once

#include "kernel.h"
#include <cuda_fp8.h>
#include <math_constants.h>
#include <kerutils/kerutils.cuh>
#include "defines.h"
#include "utils.h"
#include "config.h"

using namespace cute;

namespace sm120::decode::sparse_fp8 {

static constexpr float MAX_INIT_VAL = -1e30;

// Simplified SM80 MMA kernel — loads Q/K/V, does MMA via CuTe gemm, stores output.
// Currently zero-fills output (placeholder) — full MMA path needs CuTe layout debugging.

template<ModelType MODEL_TYPE, int NUM_HEADS>
template<typename TMAParams>
__device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(
    const SparseAttnDecodeParams &params,
    const TMAParams &tma_params)
{
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
    const int head_block_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    int start_head_idx = head_block_idx * BLOCK_M;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    for (int batch_idx = 0; batch_idx < params.b; batch_idx++) {
        bf16* gO = (bf16*)params.out + batch_idx*params.stride_o_b
                 + s_q_idx*params.stride_o_s_q + start_head_idx*params.stride_o_h_q;

        // Zero output (placeholder — full MMA path to be completed)
        for (int r = 0; r < BLOCK_M; r++) {
            for (int c = threadIdx.x; c < HEAD_DIM_V; c += NUM_THREADS)
                gO[r * params.stride_o_h_q + c] = bf16(0.0f);
        }
    }
    __syncthreads();
#endif
}

// Free __global__ kernel wrapper
template<ModelType MODEL_TYPE, int NUM_HEADS, typename TMAParams>
__global__ void sm120_global_kernel(
    __grid_constant__ const SparseAttnDecodeParams params,
    __grid_constant__ const TMAParams tma_params)
{
    KernelTemplate<MODEL_TYPE, NUM_HEADS>::template devfunc<TMAParams>(params, tma_params);
}

// Host launch
template<ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const SparseAttnDecodeParams &params) {
    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q, params.b);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((bf16*)params.q),
            make_layout(shape_Q, make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b))),
        SmemLayoutQPass{});
    CUtensorMap tensor_map_o{};
    TmaParams<decltype(shape_Q), decltype(tma_Q)> tma_params = {shape_Q, tma_Q, tensor_map_o};

    auto kernel_fn = &sm120_global_kernel<MODEL_TYPE, NUM_HEADS, decltype(tma_params)>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute((void*)kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid(NUM_HEADS / 64, params.s_q, 1);
    dim3 block(NUM_THREADS, 1, 1);
    kernel_fn<<<grid, block, smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_sm120_sparse_decode_kernel(const SparseAttnDecodeParams &params) {
    KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(params);
}

}  // namespace sm120::decode::sparse_fp8
