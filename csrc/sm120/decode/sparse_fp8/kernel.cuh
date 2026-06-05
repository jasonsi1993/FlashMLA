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
        // O accumulator: 16 rows x 512 cols per warp, 4 floats per 16x8 mma tile = 256 floats/thread
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

            // rP: 64 floats per thread for QK(16x64): 64/8=8 N-steps x 8 vals = 64?
            // Actually each mma.sync produces 4 floats per thread covering 16x8.
            // 8 N-steps x 4 = 32 floats per thread for 64 cols.
            // But each thread has 2 row-groups x 4 cols = 8 values per N-step? No, 4 per step.
            float rP[32]; for (int i = 0; i < 32; i++) rP[i] = 0.0f;

            // ---- QK passes ----
            for (int pass = 0; pass < NUM_PASSES; pass++) {
                int p_start = pass * QK_TILES_PER_PASS;
                int p_end = min(p_start + QK_TILES_PER_PASS, TOTAL_QK_TILES);
                int p_tiles = p_end - p_start;
                int p_dim = p_tiles * 64;

                // Cooperative Q load element-by-element
                bf16* gQp = (bf16*)params.q + batch_idx*params.stride_q_b
                          + s_q_idx*params.stride_q_s_q + start_head_idx*params.stride_q_h_q + p_start*64;
                for (int i = threadIdx.x; i < BLOCK_M * p_dim; i += NUM_THREADS) {
                    int r = i / p_dim, c = i % p_dim;
                    plan.q.data()[i] = gQp[r * params.stride_q_h_q + c];
                }
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
                            if constexpr (MODEL_TYPE == ModelType::V32) {
                                for (int si=0; si<4; si++)
                                    sf[si] = __ldg((const float*)(gK + HEAD_DIM_NOPE) + si);
                            } else { for (int si=0;si<4;si++) sf[si] = 1.0f; }

                            for (int dt = p_start; dt < p_end; dt++) {
                                int ld = dt - p_start;
                                fp8x16 src;
                                for (int bi=0; bi<8; bi++) {
                                    ((uint8_t*)&src.lo)[bi] = gK[dt*64 + bi];
                                    ((uint8_t*)&src.hi)[bi] = gK[dt*64 + 8 + bi];
                                }
                                bf16 sc = (bf16)sf[MODEL_TYPE==ModelType::V32 ? dt/2 : dt];
                                bf16x8 lo = cvt_fp8x8_bf16x8(src.lo, __bfloat162bfloat162(*(__nv_bfloat16*)&sc));
                                bf16x8 hi = cvt_fp8x8_bf16x8(src.hi, __bfloat162bfloat162(*(__nv_bfloat16*)&sc));
                                for (int bi = 0; bi < 8; bi++) {
                                    row[(ld*64 + bi)*TOPK_BLOCK_SIZE] = ((bf16*)&lo)[bi];
                                    row[(ld*64 + 8 + bi)*TOPK_BLOCK_SIZE] = ((bf16*)&hi)[bi];
                                }
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
                        // B is (K=16, N=8): thread loads K-rows [lane%8, lane%8+8], N-cols [lane/8, lane/8+4]
                        int bk0 = lane_id % 8, bk1 = bk0 + 8;
                        int bn0 = lane_id / 8, bn1 = bn0 + 4;
                        *reinterpret_cast<__nv_bfloat162*>(&b_regs[0]) =
                            *reinterpret_cast<__nv_bfloat162*>(sK_ptr + (ns*8+bn0)*TOPK_BLOCK_SIZE + ks*16 + bk0);
                        *reinterpret_cast<__nv_bfloat162*>(&b_regs[1]) =
                            *reinterpret_cast<__nv_bfloat162*>(sK_ptr + (ns*8+bn1)*TOPK_BLOCK_SIZE + ks*16 + bk1);

                        float c[4]; int pb = ns * 4;
                        c[0]=rP[pb]; c[1]=rP[pb+1]; c[2]=rP[pb+2]; c[3]=rP[pb+3];
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                            : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                              "r"(b_regs[0]), "r"(b_regs[1]));
                        rP[pb]=c[0]; rP[pb+1]=c[1]; rP[pb+2]=c[2]; rP[pb+3]=c[3];
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
            // Corrected register layout:
            //   rP[ns*4+0]: S[row0, ns*8 + lane/8]
            //   rP[ns*4+1]: S[row0, ns*8 + lane/8 + 4]
            //   rP[ns*4+2]: S[row1, ns*8 + lane/8]
            //   rP[ns*4+3]: S[row1, ns*8 + lane/8 + 4]
            //   row0 = mm_row + lane%8, row1 = row0 + 8
            bf16* sS_ptr = plan.s.data();
            int sr0 = mm_row + (lane_id % 8);
            int sr1 = sr0 + 8;
            for (int ns = 0; ns < TOPK_BLOCK_SIZE/8; ns++) {
                int pb = ns * 4;
                int sc0 = ns * 8 + (lane_id / 8);
                int sc1 = sc0 + 4;
                if (sc0 < TOPK_BLOCK_SIZE) {
                    sS_ptr[sr0 * TOPK_BLOCK_SIZE + sc0] = (bf16)rP[pb];
                    sS_ptr[sr0 * TOPK_BLOCK_SIZE + sc1] = (bf16)rP[pb + 1];
                    sS_ptr[sr1 * TOPK_BLOCK_SIZE + sc0] = (bf16)rP[pb + 2];
                    sS_ptr[sr1 * TOPK_BLOCK_SIZE + sc1] = (bf16)rP[pb + 3];
                }
            }
            __syncthreads();

            // ---- PV: S x V -> O ----
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
                                for (int si=0; si<4; si++) sf[si] = ((const float*)(gK + HEAD_DIM_NOPE))[si];
                            else { for(int si=0;si<4;si++) sf[si]=1.0f; }
                            int vo = vh*HV;
                            for (int vi = 0; vi < HV/16; vi++) {
                                int vd = vo + vi*16;
                                fp8x16 src;
                                *reinterpret_cast<uint2*>(&src.lo) = *reinterpret_cast<const uint2*>(gK + vd);
                                *reinterpret_cast<uint2*>(&src.hi) = *reinterpret_cast<const uint2*>(gK + vd + 8);
                                bf16 sc = vd/128<4 ? (bf16)sf[vd/128] : (bf16)1.0f;
                                bf16x8 lo = cvt_fp8x8_bf16x8(src.lo, __bfloat162bfloat162(*(__nv_bfloat16*)&sc));
                                bf16x8 hi = cvt_fp8x8_bf16x8(src.hi, __bfloat162bfloat162(*(__nv_bfloat16*)&sc));
                                for (int bi=0; bi<8; bi++) {
                                    vrow[(vi*16 + bi)*TOPK_BLOCK_SIZE] = ((bf16*)&lo)[bi];
                                    vrow[(vi*16 + 8 + bi)*TOPK_BLOCK_SIZE] = ((bf16*)&hi)[bi];
                                }
                            }
                        } else {
                            for (int i = 0; i < HV; i++) vrow[i*TOPK_BLOCK_SIZE] = bf16(0.0f);
                        }
                    }
                }
                __syncthreads();

                // Manual mma.sync PV loop: S[16x64] x V[64x256] -> O[16x256]
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

                        // 4 outputs per N-step, 128 per V-half -> 256 total per thread
                        int ob = ns * 4 + vh * 128;
                        float c[4];
                        c[0]=rO[ob]; c[1]=rO[ob+1]; c[2]=rO[ob+2]; c[3]=rO[ob+3];
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                            : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                              "r"(b_regs[0]), "r"(b_regs[1]));
                        rO[ob]=c[0]; rO[ob+1]=c[1]; rO[ob+2]=c[2]; rO[ob+3]=c[3];
                    }
                }
                __syncthreads();
            }
        }  // K/V blocks

        // ---- Store output: rO -> global memory ----
        // Corrected: row=lane%8/+8, col=lane/8/+4
        bf16* gO = (bf16*)params.out + batch_idx*params.stride_o_b
                 + s_q_idx*params.stride_o_s_q + start_head_idx*params.stride_o_h_q;
        int max_row = min(params.h_q - start_head_idx, BLOCK_M);
        int or0 = mm_row + (lane_id % 8);
        int or1 = or0 + 8;
        for (int ns = 0; ns < HEAD_DIM_V/8; ns++) {
            int ob = ns * 4;
            int oc0 = ns * 8 + (lane_id / 8);
            int oc1 = oc0 + 4;
            if (oc0 < HEAD_DIM_V) {
                if (or0 < max_row) {
                    gO[or0 * params.stride_o_h_q + oc0] = (bf16)rO[ob];
                    gO[or0 * params.stride_o_h_q + oc1] = (bf16)rO[ob + 1];
                }
                if (or1 < max_row) {
                    gO[or1 * params.stride_o_h_q + oc0] = (bf16)rO[ob + 2];
                    gO[or1 * params.stride_o_h_q + oc1] = (bf16)rO[ob + 3];
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

// Dummy TMA params (kernel uses cooperative copy, not TMA)
struct DummyTmaParams {};

template<ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const SparseAttnDecodeParams &params) {
    DummyTmaParams tma_params{};
    auto kernel_fn = &sm120_global_kernel<MODEL_TYPE, NUM_HEADS, DummyTmaParams>;
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
