#pragma once

#include "kernel.h"

#include <cuda_fp8.h>
#include <math_constants.h>
#include <cutlass/arch/barrier.h>

#include <kerutils/kerutils.cuh>

#include "utils.h"
#include "sm90/decode/sparse_fp8/components/dequant.h"
#include "sm90/decode/sparse_fp8/components/helpers.h"
#include "config.h"

using namespace cute;

namespace sm120::decode::sparse_fp8 {

static constexpr float MAX_INIT_VAL = -1e30;
using fp8_e8m0 = __nv_fp8_e8m0;

// Softmax: P (registers) → S (shared memory), update running max/sum
template<typename Tensor0, typename Tensor1, typename Tensor2>
__forceinline__ __device__ void scale_softmax_sm120(
    Tensor0 &rP,
    Tensor1 &rS,
    Tensor2 &rO,
    float scale_softmax_log2,
    float sScale[],
    float rM[2],
    float rL[2],
    bool is_kv_valid[],
    int block_idx,
    int idx_in_warpgroup)
{
    float scale_for_olds[2];
    CUTE_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        Tensor cur_rP = flatten(rP(make_coord(_, local_row_idx, _), _, _));
        Tensor cur_rS = flatten(rS(make_coord(_, local_row_idx, _), _, _));
        Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));

        float cur_max = -INFINITY;
        CUTE_UNROLL
        for (int i = 0; i < size(cur_rP); ++i) {
            if (!is_kv_valid[(i&1)+(i/2)*8+(idx_in_warpgroup%4)*2])
                cur_rP(i) = -INFINITY;
            cur_max = max(cur_max, cur_rP(i));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

        cur_max *= scale_softmax_log2;
        float old_max = rM[local_row_idx];
        rM[local_row_idx] = max(cur_max, old_max);
        float scale_for_old = exp2f(old_max - rM[local_row_idx]);
        scale_for_olds[local_row_idx] = scale_for_old;

        CUTE_UNROLL
        for (int i = 0; i < size(cur_rO); ++i) {
            cur_rO(i) *= scale_for_old;
        }

        float cur_sum = 0;
        CUTE_UNROLL
        for (int i = 0; i < size(cur_rP); ++i) {
            cur_rP(i) = exp2f(cur_rP(i)*scale_softmax_log2 - rM[local_row_idx]);
            cur_rS(i) = (bf16)cur_rP(i);
            cur_sum += cur_rP(i);
        }

        rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum;
    }
    if (idx_in_warpgroup%4 == 0)
        *(float2*)(sScale + 2*(idx_in_warpgroup/4)) = *(float2*)(scale_for_olds);
}

template<ModelType MODEL_TYPE, int NUM_HEADS, typename TMAParams>
__global__ void sm120_sparse_decode_kernel(
    __grid_constant__ const SparseAttnDecodeParams params,
    __grid_constant__ const TMAParams tma_params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)) || defined(__CLION_IDE__) || defined(__VSCODE_IDE__)
    const int head_block_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int idx_in_warpgroup = threadIdx.x % 128;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQPass{});
    Tensor sK = make_tensor(make_smem_ptr(plan.k.data()), SmemLayoutKPass{});
    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
    Tensor sOBuf = make_tensor(make_smem_ptr(plan.oBuf.data()), SmemLayoutOBuf{});

    float* sM = plan.sM;
    float* sL = plan.sL;
    float* sScale = plan.sScale;

    // Prefetch TMA
    if (threadIdx.x == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_o);
    }
    __syncthreads();

    // Tiled MMA setup
    TiledMMA tiled_mma = TiledMMA{};
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);

    // rP fragment for QK (64 × 64)
    Tensor rP = partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});

    // Count valid query rows
    int num_valid_seq_q = min(params.h_q - head_block_idx*BLOCK_M, BLOCK_M);
    int start_head_idx = head_block_idx * BLOCK_M;

    // Batch loop
    static constexpr int PASS_DIM = QK_TILES_PER_PASS * 64;

    for (int batch_idx = 0; batch_idx < params.b; batch_idx++) {
        float rM[2] = {MAX_INIT_VAL, MAX_INIT_VAL};
        float rL[2] = {0.0f, 0.0f};

        // rO accumulator
        Tensor rO = partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V/2>>{});
        cute::fill(rO, 0.f);

        // attn_sink
        float rAttn_sink[2] = {-CUDART_INF_F, -CUDART_INF_F};
        if (params.attn_sink != nullptr) {
            for (int i = 0; i < 2; ++i) {
                int head_idx = start_head_idx + (idx_in_warpgroup / 4) * 8 + (idx_in_warpgroup % 4);
                if (head_idx < params.h_q)
                    rAttn_sink[i] = __ldg(params.attn_sink + head_idx) * CUDART_L2E_F;
            }
        }

        int total_blocks = params.topk / TOPK_BLOCK_SIZE;

        // Process K/V blocks
        for (int block_idx = 0; block_idx < total_blocks; block_idx++) {
            // === QK pass loop ===
            cute::fill(rP, 0.f);

            // Compute is_kv_valid for this block
            int* gIndices = params.indices + batch_idx*params.stride_indices_b
                          + s_q_idx*params.stride_indices_s_q
                          + block_idx*TOPK_BLOCK_SIZE;
            bool is_kv_valid[TOPK_BLOCK_SIZE];
            for (int i = threadIdx.x; i < TOPK_BLOCK_SIZE; i += NUM_THREADS) {
                int idx = __ldg(gIndices + i);
                is_kv_valid[i] = (idx != -1);
            }
            __syncthreads();

            static constexpr int TOTAL_QK_TILES = HEAD_DIM_K / 64;
            static constexpr int NUM_PASSES = (TOTAL_QK_TILES + QK_TILES_PER_PASS - 1) / QK_TILES_PER_PASS;

            for (int qk_pass = 0; qk_pass < NUM_PASSES; qk_pass++) {
                int pass_start_tile = qk_pass * QK_TILES_PER_PASS;
                int pass_end_tile = min(pass_start_tile + QK_TILES_PER_PASS, TOTAL_QK_TILES);
                int pass_tiles = pass_end_tile - pass_start_tile;

                // Load Q pass slice via TMA
                if (threadIdx.x == 0) {
                    Tensor gQ = flat_divide(
                        tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, batch_idx),
                        Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
                    )(_, _, head_block_idx, _0{});
                    Tensor gQ_pass = local_tile(gQ,
                        Shape<Int<BLOCK_M>, Int<PASS_DIM>>{},
                        make_coord(_0{}, Int<pass_start_tile * 64>{})
                    );
                    // Simple async copy for Q pass
                    int q_rows = BLOCK_M;
                    int q_cols = pass_tiles * 64;
                    for (int r = 0; r < q_rows; r++) {
                        for (int c = 0; c < q_cols; c += 8) {
                            *reinterpret_cast<uint4*>(plan.q.data() + r * q_cols + c) =
                                *reinterpret_cast<const uint4*>(
                                    (bf16*)params.q
                                    + batch_idx*params.stride_q_b
                                    + s_q_idx*params.stride_q_s_q
                                    + (start_head_idx + r)*params.stride_q_h_q
                                    + pass_start_tile*64 + c);
                        }
                    }
                }
                __syncthreads();

                // Load K pass slice via dequantization (simplified — just zero for now)
                // In full implementation, dequantize FP8→BF16 from KV cache
                // For now, zero-fill K to test the MMA path
                if (threadIdx.x == 0) {
                    for (int i = 0; i < TOPK_BLOCK_SIZE * pass_tiles * 64; i++) {
                        plan.k.data()[i] = bf16(0.0f);
                    }
                }
                __syncthreads();

                // Create K tensor for this pass
                auto k_layout_pass = Layout<Shape<Int<TOPK_BLOCK_SIZE>, Int<pass_tiles*64>>,
                                            Stride<_1, Int<TOPK_BLOCK_SIZE>>>{};
                Tensor sK_pass = make_tensor(make_smem_ptr(plan.k.data()), k_layout_pass);

                // QK MMA
                gemm(tiled_mma,
                     thr_mma.partition_fragment_A(sQ(_, Int<0>{}, Int<pass_tiles*64>{})),
                     thr_mma.partition_fragment_B(sK_pass),
                     rP);
            }

            // Softmax on accumulated rP
            Tensor rS_sm120 = make_tensor<bf16>(partition_shape_A(tiled_mma, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));
            {
                float dummy_scale[2];
                scale_softmax_sm120(rP, rS_sm120, rO, params.sm_scale_div_log2,
                                    sScale, rM, rL, is_kv_valid, block_idx, idx_in_warpgroup);
            }

            // Store S to shared
            {
                auto r2s_copy = make_tiled_copy_S(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, bf16>{}, tiled_mma);
                ThrCopy thr_copy = r2s_copy.get_slice(threadIdx.x);
                cute::copy(r2s_copy, thr_copy.retile_S(rS_sm120), thr_copy.partition_D(sS));
            }
            __syncthreads();

            // === PV ===
            // Process V in two halves (256 dims each)
            static constexpr int HALF_V_DIM = HEAD_DIM_V / 2;
            auto v_layout = Layout<Shape<Int<TOPK_BLOCK_SIZE>, Int<HALF_V_DIM>>,
                                   Stride<_1, Int<TOPK_BLOCK_SIZE>>>{};
            Tensor sV_half = make_tensor(make_smem_ptr(plan.k.data()), v_layout);

            // First V half — zero for now
            if (threadIdx.x == 0) {
                for (int i = 0; i < TOPK_BLOCK_SIZE * HALF_V_DIM; i++) {
                    plan.k.data()[i] = bf16(0.0f);
                }
            }
            __syncthreads();

            gemm(tiled_mma,
                 thr_mma.partition_fragment_A(sS),
                 thr_mma.partition_fragment_B(sV_half),
                 rO);

            // Second V half
            if (threadIdx.x == 0) {
                for (int i = 0; i < TOPK_BLOCK_SIZE * HALF_V_DIM; i++) {
                    plan.k.data()[i] = bf16(0.0f);
                }
            }
            __syncthreads();

            gemm(tiled_mma,
                 thr_mma.partition_fragment_A(sS),
                 thr_mma.partition_fragment_B(sV_half),
                 rO);
        }

        // === Store output ===
        // Write rO to sOBuf in BF16, then TMA store to global
        {
            float o_scales[2] = {1.0f, 1.0f};
            for (int idx = 0; idx < size(rO); idx += 2) {
                int row = (idx_in_warpgroup / 4) * 8 + (idx_in_warpgroup % 4);
                int col = idx / 4 * 8 + (idx % 4) * 2;
                if (row < num_valid_seq_q && col < HEAD_DIM_V) {
                    sOBuf(row, col) = (bf16)(rO(idx) * o_scales[idx % 4 >= 2]);
                    sOBuf(row, col + 1) = (bf16)(rO(idx + 1) * o_scales[(idx + 1) % 4 >= 2]);
                }
            }
        }
        __syncthreads();

        // TMA store
        if (threadIdx.x == 0) {
            SM90_TMA_STORE_5D::copy(
                &tma_params.tensor_map_o,
                plan.oBuf.data(),
                0, head_block_idx*BLOCK_M, 0,
                s_q_idx, batch_idx
            );
            cute::tma_store_arrive();
        }
        cute::tma_store_wait<0>();
        __syncthreads();
    }
#else
    // Non-SM80+ arch fallback
#endif
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const SparseAttnDecodeParams &params) {
    (void)params;
    // TODO: Implement SM80-MMA kernel launch
    // For now, this is a placeholder — the kernel is work-in-progress
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_sm120_sparse_decode_kernel(const SparseAttnDecodeParams &params) {
    KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(params);
}

}  // namespace sm120::decode::sparse_fp8
