#pragma once

#include "kernel.h"
#include <kerutils/kerutils.cuh>
#include "defines.h"
#include "utils.h"
#include "sm90/decode/sparse_fp8/components/dequant.h"
#include "config.h"

namespace sm120::decode::sparse_fp8 {
using sm90::decode::sparse_fp8::fp8x16;
using sm90::decode::sparse_fp8::cvt_fp8x8_bf16x8;

static constexpr float MAX_INIT_VAL = -1e30;

template<ModelType MODEL_TYPE, int NUM_HEADS>
template<typename TMAParams>
__device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(
    const SparseAttnDecodeParams &params, const TMAParams &tma_params)
{
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
    const int head_block_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    int start_head_idx = head_block_idx * BLOCK_M;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    int mm_row = warp_id * 16;

    static constexpr int TOTAL_QK_TILES = HEAD_DIM_K / 64;
    static constexpr int NUM_PASSES = (TOTAL_QK_TILES + QK_TILES_PER_PASS - 1) / QK_TILES_PER_PASS;
    static constexpr int PASS_DIM_S = QK_TILES_PER_PASS * 64;  // 320

    for (int batch_idx = 0; batch_idx < params.b; batch_idx++) {
        float rM[2] = {MAX_INIT_VAL, MAX_INIT_VAL}, rL[2] = {0, 0};
        // O accumulator: 16 rows × 512 cols per warp, 4 floats per 16×8 mma tile = 256 floats/thread
        float rO[256]; for (int i = 0; i < 256; i++) rO[i] = 0.0f;
        float o_scales_pre[256]; for (int i = 0; i < 256; i++) o_scales_pre[i] = 1.0f;

        int total_blocks = params.topk / TOPK_BLOCK_SIZE;
        for (int block_idx = 0; block_idx < total_blocks; block_idx++) {
            // is_kv_valid
            bool kv_valid[TOPK_BLOCK_SIZE];
            int* gIdx = params.indices + batch_idx*params.stride_indices_b
                       + s_q_idx*params.stride_indices_s_q + block_idx*TOPK_BLOCK_SIZE;
            for (int i = threadIdx.x; i < TOPK_BLOCK_SIZE; i += NUM_THREADS)
                kv_valid[i] = (__ldg(gIdx + i) != -1);
            __syncthreads();

            // rP: 64 floats per thread for QK(16×64): 64/8=8 N-steps × 8 vals = 64?
            // Actually each mma.sync produces 4 floats per thread covering 16×8.
            // 8 N-steps × 4 = 32 floats per thread for 64 cols.
            // But each thread has 2 row-groups × 4 cols = 8 values per N-step? No, 4 per step.
            float rP[32]; for (int i = 0; i < 32; i++) rP[i] = 0.0f;

            // ---- QK passes ----
            for (int pass = 0; pass < NUM_PASSES; pass++) {
                int p_start = pass * QK_TILES_PER_PASS;
                int p_end = min(p_start + QK_TILES_PER_PASS, TOTAL_QK_TILES);
                int p_tiles = p_end - p_start;
                int p_dim = p_tiles * 64;

                // Cooperative Q load (row-major, 64 × p_dim)
                bf16* gQp = (bf16*)params.q + batch_idx*params.stride_q_b
                          + s_q_idx*params.stride_q_s_q + start_head_idx*params.stride_q_h_q + p_start*64;
                for (int r = threadIdx.x; r < BLOCK_M; r += NUM_THREADS)
                    for (int c = 0; c < p_dim; c += 8)
                        *reinterpret_cast<uint4*>(plan.q.data() + r*p_dim + c) =
                            *reinterpret_cast<const uint4*>(gQp + r*params.stride_q_h_q + c);
                __syncthreads();

                // K dequant from FP8 cache: see sparse_fp8/../splitkv_mla.cuh producer for reference
                {
                    int toff = block_idx * TOPK_BLOCK_SIZE;
                    for (int t = threadIdx.x; t < TOPK_BLOCK_SIZE; t += NUM_THREADS) {
                        int tok = __ldg(params.indices + batch_idx*params.stride_indices_b
                                       + s_q_idx*params.stride_indices_s_q + toff + t);
                        bf16* row = plan.k.data() + t;
                        if (tok != -1) {
                            int blk = tok / params.page_block_size;
                            int rel = tok % params.page_block_size;
                            fp8* gK = (fp8*)params.kv + blk*params.stride_kv_block + rel*params.stride_kv_row;
                            float sf[4];
                            if constexpr (MODEL_TYPE == ModelType::V32)
                                *(float4*)sf = *(const float4*)(gK + HEAD_DIM_NOPE);
                            else { for (int si=0;si<4;si++) sf[si] = 1.0f; }

                            for (int dt = p_start; dt < p_end; dt++) {
                                int ld = dt - p_start;
                                fp8x16 src = *(const fp8x16*)(gK + dt*64);
                                bf16 sc = (bf16)sf[MODEL_TYPE==ModelType::V32 ? dt/2 : dt];
                                bf16x8 lo = cvt_fp8x8_bf16x8(src.lo, __bfloat162bfloat162(*(__nv_bfloat16*)&sc));
                                bf16x8 hi = cvt_fp8x8_bf16x8(src.hi, __bfloat162bfloat162(*(__nv_bfloat16*)&sc));
                                *(bf16x8*)(row + ld*64*TOPK_BLOCK_SIZE) = lo;
                                *(bf16x8*)(row + (ld*64+8)*TOPK_BLOCK_SIZE) = hi;
                            }
                        } else {
                            for (int i = 0; i < p_dim; i++) row[i*TOPK_BLOCK_SIZE] = bf16(0.0f);
                        }
                    }
                }
                __syncthreads();

                // Manual mma.sync QK loop
                bf16* sQ_ptr = plan.q.data();
                bf16* sK_ptr = plan.k.data();

                for (int ks = 0; ks < p_dim/16; ks++) {
                    unsigned a_regs[4];
                    int ar0 = lane_id / 4, ar1 = ar0 + 8, ac = (lane_id % 4) * 2;
                    for (int g = 0; g < 2; g++) {
                        int r = (g == 0 ? ar0 : ar1);
                        *reinterpret_cast<__nv_bfloat162*>(&a_regs[g*2]) =
                            *reinterpret_cast<__nv_bfloat162*>(sQ_ptr + (mm_row+r)*p_dim + ks*16 + ac);
                        *reinterpret_cast<__nv_bfloat162*>(&a_regs[g*2+1]) =
                            *reinterpret_cast<__nv_bfloat162*>(sQ_ptr + (mm_row+r)*p_dim + ks*16 + ac+1);
                    }
                    for (int ns = 0; ns < TOPK_BLOCK_SIZE/8; ns++) {
                        unsigned b_regs[2];
                        int br = (lane_id%4)*2, bc0 = lane_id/4, bc1 = bc0+8;
                        *reinterpret_cast<__nv_bfloat162*>(&b_regs[0]) =
                            *reinterpret_cast<__nv_bfloat162*>(sK_ptr + (ns*8+br)*TOPK_BLOCK_SIZE + ks*16 + bc0);
                        *reinterpret_cast<__nv_bfloat162*>(&b_regs[1]) =
                            *reinterpret_cast<__nv_bfloat162*>(sK_ptr + (ns*8+br)*TOPK_BLOCK_SIZE + ks*16 + bc1);

                        float c[4]; int pb = ns*2;
                        c[0]=rP[pb]; c[1]=rP[pb+1]; c[2]=rP[pb+16]; c[3]=rP[pb+17];
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                            : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                              "r"(b_regs[0]), "r"(b_regs[1]));
                        rP[pb]=c[0]; rP[pb+1]=c[1]; rP[pb+16]=c[2]; rP[pb+17]=c[3];
                    }
                }
                __syncthreads();
            }  // QK passes

            // ---- Online softmax ----
            // Scale previous O by the ratio of old/new max; accumulate new P
            {
                float scale_old[2];
                for (int lr = 0; lr < 2; lr++) {
                    int row_base = lr * 8 * 2;  // 16 elements per half-warp-row
                    // Find max in this row's rP values
                    float cm = -INFINITY;
                    for (int ns = 0; ns < TOPK_BLOCK_SIZE/8; ns++) {
                        int pb = ns*2 + lr*16;
                        // rP[pb], rP[pb+1]: valid only if kv_valid
                        int col_base = ns*8;
                        for (int ci = 0; ci < 2; ci++) {
                            int col = col_base + (lane_id%4)*2 + ci;
                            if (kv_valid[col]) cm = fmaxf(cm, rP[pb+ci]);
                            // The other half of registers (pb+16) is for row+8, same columns
                            if (kv_valid[col]) cm = fmaxf(cm, rP[pb+16+ci]);
                        }
                    }
                    for (int s = 1; s < 4; s *= 2) cm = fmaxf(cm, __shfl_xor_sync(0xffffffff, cm, s));
                    cm *= params.sm_scale_div_log2;
                    float om = rM[lr]; rM[lr] = fmaxf(cm, om);
                    scale_old[lr] = exp2f(om - rM[lr]);

                    // Rescale O
                    for (int i = lr*128; i < lr*128+128; i++) rO[i] *= scale_old[lr];

                    // Softmax and accumulate
                    float cs = 0;
                    for (int ns = 0; ns < TOPK_BLOCK_SIZE/8; ns++) {
                        int pb = ns*2 + lr*16;
                        for (int ci = 0; ci < 2; ci++) {
                            int col = ns*8 + (lane_id%4)*2 + ci;
                            if (kv_valid[col]) {
                                float pv = exp2f(rP[pb+ci]*params.sm_scale_div_log2 - rM[lr]);
                                rP[pb+ci] = pv;
                                cs += pv;
                                pv = exp2f(rP[pb+16+ci]*params.sm_scale_div_log2 - rM[lr]);
                                rP[pb+16+ci] = pv;
                                cs += pv;
                            } else {
                                rP[pb+ci] = 0; rP[pb+16+ci] = 0;
                            }
                        }
                    }
                    rL[lr] = rL[lr]*scale_old[lr] + cs;
                }
            }

            // Store S to shared memory (needed for PV)
            bf16* sS_ptr = plan.s.data();
            for (int ns = 0; ns < TOPK_BLOCK_SIZE/8; ns++) {
                int pb = ns*2;
                int col_base = ns*8;
                for (int ci = 0; ci < 2; ci++) {
                    int col = col_base + (lane_id%4)*2 + ci;
                    if (col < TOPK_BLOCK_SIZE) {
                        sS_ptr[(mm_row + (lane_id/4)) * TOPK_BLOCK_SIZE + col] = (bf16)rP[pb+ci];
                        sS_ptr[(mm_row + (lane_id/4) + 8) * TOPK_BLOCK_SIZE + col] = (bf16)rP[pb+16+ci];
                    }
                }
            }
            __syncthreads();

            // ---- PV: S × V → O ----
            for (int vh = 0; vh < 2; vh++) {
                static constexpr int HV = HEAD_DIM_V / 2;
                // Load V from FP8 cache into k buffer
                {
                    int toff = block_idx * TOPK_BLOCK_SIZE;
                    for (int t = threadIdx.x; t < TOPK_BLOCK_SIZE; t += NUM_THREADS) {
                        int tok = __ldg(params.indices + batch_idx*params.stride_indices_b
                                       + s_q_idx*params.stride_indices_s_q + toff + t);
                        bf16* vrow = plan.k.data() + t;
                        if (tok != -1) {
                            int blk = tok / params.page_block_size;
                            int rel = tok % params.page_block_size;
                            fp8* gK = (fp8*)params.kv + blk*params.stride_kv_block + rel*params.stride_kv_row;
                            float sf[4];
                            if constexpr (MODEL_TYPE==ModelType::V32)
                                *(float4*)sf = *(const float4*)(gK + HEAD_DIM_NOPE);
                            else { for(int si=0;si<4;si++) sf[si]=1.0f; }
                            int vo = vh*HV;
                            for (int vi = 0; vi < HV/16; vi++) {
                                int vd = vo + vi*16;
                                fp8x16 src = *(const fp8x16*)(gK + vd);
                                bf16 sc = vd/128<4 ? (bf16)sf[vd/128] : (bf16)1.0f;
                                bf16x8 lo = cvt_fp8x8_bf16x8(src.lo, __bfloat162bfloat162(*(__nv_bfloat16*)&sc));
                                bf16x8 hi = cvt_fp8x8_bf16x8(src.hi, __bfloat162bfloat162(*(__nv_bfloat16*)&sc));
                                *(bf16x8*)(vrow + vi*16*TOPK_BLOCK_SIZE) = lo;
                                *(bf16x8*)(vrow + (vi*16+8)*TOPK_BLOCK_SIZE) = hi;
                            }
                        } else {
                            for (int i = 0; i < HV; i++) vrow[i*TOPK_BLOCK_SIZE] = bf16(0.0f);
                        }
                    }
                }
                __syncthreads();

                // Manual mma.sync PV loop: S[16×64] × V[64×256] → O[16×256]
                bf16* sS_ptr2 = plan.s.data();
                bf16* sV_ptr = plan.k.data();

                for (int ks = 0; ks < TOPK_BLOCK_SIZE/16; ks++) {
                    // Load A: S[mm:16, ks*16:(ks+1)*16] from shared
                    unsigned a_regs[4];
                    int ar0 = lane_id/4, ar1 = ar0+8, ac = (lane_id%4)*2;
                    for (int g = 0; g < 2; g++) {
                        int r = (g==0 ? ar0 : ar1);
                        *reinterpret_cast<__nv_bfloat162*>(&a_regs[g*2]) =
                            *reinterpret_cast<__nv_bfloat162*>(sS_ptr2 + (mm_row+r)*TOPK_BLOCK_SIZE + ks*16 + ac);
                        *reinterpret_cast<__nv_bfloat162*>(&a_regs[g*2+1]) =
                            *reinterpret_cast<__nv_bfloat162*>(sS_ptr2 + (mm_row+r)*TOPK_BLOCK_SIZE + ks*16 + ac+1);
                    }
                    for (int ns = 0; ns < HV/8; ns++) {
                        unsigned b_regs[2];
                        int br = (lane_id%4)*2, bc0 = lane_id/4, bc1 = bc0+8;
                        *reinterpret_cast<__nv_bfloat162*>(&b_regs[0]) =
                            *reinterpret_cast<__nv_bfloat162*>(sV_ptr + (ns*8+br)*TOPK_BLOCK_SIZE + ks*16 + bc0);
                        *reinterpret_cast<__nv_bfloat162*>(&b_regs[1]) =
                            *reinterpret_cast<__nv_bfloat162*>(sV_ptr + (ns*8+br)*TOPK_BLOCK_SIZE + ks*16 + bc1);

                        int ob = ns*2 + vh*(HV/8 * 2);
                        float c[4];
                        c[0]=rO[ob]; c[1]=rO[ob+1]; c[2]=rO[ob+16]; c[3]=rO[ob+17];
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                            : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                              "r"(b_regs[0]), "r"(b_regs[1]));
                        rO[ob]=c[0]; rO[ob+1]=c[1]; rO[ob+16]=c[2]; rO[ob+17]=c[3];
                    }
                }
                __syncthreads();
            }
        }  // K/V blocks

        // ---- Store output: rO → global memory ----
        bf16* gO = (bf16*)params.out + batch_idx*params.stride_o_b
                 + s_q_idx*params.stride_o_s_q + start_head_idx*params.stride_o_h_q;
        for (int ns = 0; ns < HEAD_DIM_V/8; ns++) {
            int vh_half = ns / (HEAD_DIM_V/16);
            int ob = ns*2;
            int col_base = ns*8;
            for (int ci = 0; ci < 2; ci++) {
                int col = col_base + (lane_id%4)*2 + ci;
                if (col < HEAD_DIM_V) {
                    int r0 = mm_row + (lane_id/4);
                    int r1 = r0 + 8;
                    if (r0 < params.h_q - start_head_idx)
                        gO[r0*params.stride_o_h_q + col] = (bf16)rO[ob+ci];
                    if (r1 < params.h_q - start_head_idx)
                        gO[r1*params.stride_o_h_q + col] = (bf16)rO[ob+16+ci];
                }
            }
        }
        __syncthreads();
    }
#endif
}

template<ModelType MODEL_TYPE, int NUM_HEADS, typename TMAParams>
__global__ void sm120_global_kernel(
    __grid_constant__ const SparseAttnDecodeParams params,
    __grid_constant__ const TMAParams tma_params)
{
    KernelTemplate<MODEL_TYPE, NUM_HEADS>::template devfunc<TMAParams>(params, tma_params);
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const SparseAttnDecodeParams &params) {
    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q, params.b);
    auto tma_Q = cute::make_tma_copy(SM90_TMA_LOAD{},
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
