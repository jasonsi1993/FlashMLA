#pragma once

#include "kernel.h"
#include <kerutils/kerutils.cuh>
#include "defines.h"
#include "utils.h"
#include "sm90/decode/sparse_fp8/components/dequant.h"
#include "config.h"
#include "debug.h"

namespace sm120::decode::sparse_fp8 {
using sm90::decode::sparse_fp8::fp8x16;
using sm90::decode::sparse_fp8::cvt_fp8x8_bf16x8;
using fp8_e8m0 = __nv_fp8_e8m0;

static constexpr float MAX_INIT_VAL = -1e30;

// QK MMA: uses CuTe for correct register layout, manual asm for MMA instruction
// Templated on the TiledMMA type to get correct per-warp partitioning
template<typename TiledMMA>
static __device__ __noinline__
void qk_mma_kernel(bf16* sQ_ptr, bf16* sK_ptr, float* rP,
                   int p_dim, int topk_blocks, int lane_id, int mm_row) {
    ThrMMA thr = TiledMMA{}.get_slice(lane_id);

    for (int ks = 0; ks < p_dim/16; ks++) {
        // Q: [16 rows, K dims] at warp offset + ks*16
        Tensor sQ = make_tensor(make_smem_ptr(sQ_ptr + mm_row * p_dim + ks * 16),
            Layout<Shape<_16, _16>, Stride<_16, _1>>{});

        // K: [64 tokens, K dims]
        Tensor sK = make_tensor(make_smem_ptr(sK_ptr + ks * 16),
            Layout<Shape<_64, _16>, Stride<_16, _1>>{});

        // CuTe partition: per-thread smem views
        Tensor tCsQ = thr.partition_A(sQ);
        Tensor tCsK = thr.partition_B(sK);

        // Load A from smem into registers via CuTe
        Tensor tCrQ = thr.make_fragment_A(tCsQ);
        cute::copy(tCsQ, tCrQ);
        unsigned a_regs[4] = {0};
        #pragma unroll
        for (int i = 0; i < 4; i++)
            a_regs[i] = reinterpret_cast<const unsigned&>(tCrQ(i, _0{}, _0{}));

        for (int ns = 0; ns < topk_blocks/8; ns++) {
            // B for this ns token group: [8 tokens, K dims]
            Tensor sK_ns = make_tensor(make_smem_ptr(sK_ptr + ns * 8 * p_dim + ks * 16),
                Layout<Shape<_8, _16>, Stride<_16, _1>>{});

            Tensor tCsK_ns = thr.partition_B(sK_ns);
            Tensor tCrK = thr.make_fragment_B(tCsK_ns);
            cute::copy(tCsK_ns, tCrK);
            unsigned b_regs[2] = {0};
            #pragma unroll
            for (int i = 0; i < 2; i++)
                b_regs[i] = reinterpret_cast<const unsigned&>(tCrK(i, _0{}, _0{}));

            float c[4] = {0, 0, 0, 0};
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                  "r"(b_regs[0]), "r"(b_regs[1]));

            int pb = ns * 4;
            rP[pb] += c[0]; rP[pb+1] += c[1]; rP[pb+2] += c[2]; rP[pb+3] += c[3];
        }
    }
}

// Isolated PV MMA: takes only pointers it needs
static __device__ __noinline__
void pv_mma_kernel(float* sS_ptr, bf16* sV_ptr, float* rO,
                   int hv, int topk_blocks, int vh, int lane_id, int mm_row) {
    for (int ks = 0; ks < topk_blocks/16; ks++) {
        unsigned a_regs[4];
        int ar0 = lane_id % 8, ar1 = ar0 + 8;
        int a_col = (lane_id / 8) * 2;
        for (int g = 0; g < 2; g++) {
            int r = (g==0 ? ar0 : ar1);
            float* a_src = sS_ptr + (mm_row+r)*topk_blocks + ks*16;
            ((bf16*)&a_regs[g*2])[0]   = (bf16)(a_src[a_col]);
            ((bf16*)&a_regs[g*2])[1]   = (bf16)(a_src[a_col + 1]);
            ((bf16*)&a_regs[g*2+1])[0] = (bf16)(a_src[a_col + 8]);
            ((bf16*)&a_regs[g*2+1])[1] = (bf16)(a_src[a_col + 9]);
        }
        for (int ns = 0; ns < hv/8; ns++) {
            unsigned b_regs[2];
            int bk0 = lane_id % 8, bk1 = bk0 + 8;
            int bn0 = lane_id / 8, bn1 = bn0 + 4;
            ((bf16*)&b_regs[0])[0] = sV_ptr[(ks*16 + bk0)*hv + (ns*8+bn0)];
            ((bf16*)&b_regs[0])[1] = sV_ptr[(ks*16 + bk0)*hv + (ns*8+bn1)];
            ((bf16*)&b_regs[1])[0] = sV_ptr[(ks*16 + bk1)*hv + (ns*8+bn0)];
            ((bf16*)&b_regs[1])[1] = sV_ptr[(ks*16 + bk1)*hv + (ns*8+bn1)];
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
}

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

    int c_col0 = lane_id / 8;  // used by softmax

    // Diagnostic: zero-init shared memory to rule out cold-start read-before-write
    for (int i = threadIdx.x; i < sizeof(SharedMemoryPlan)/4; i += NUM_THREADS) {
        ((int*)wksp_buf)[i] = 0;
    }
    __threadfence_block(); __syncthreads(); asm volatile("" ::: "memory");

    // Copy hot params to shared memory to avoid __grid_constant__ codegen issues
    __shared__ int smem_b;
    __shared__ int smem_topk;
    if (threadIdx.x == 0) { smem_b = params.b; smem_topk = params.topk; }
    __threadfence_block(); __syncthreads(); asm volatile("" ::: "memory");


    for (int batch_idx = 0; batch_idx < smem_b; batch_idx++) {
        float rM[2] = {MAX_INIT_VAL, MAX_INIT_VAL}, rL[2] = {0, 0};
        // O accumulator: 16 rows x 512 cols per warp, 4 floats per 16x8 mma tile = 256 floats/thread
        float rO[256]; for (int i = 0; i < 256; i++) rO[i] = 0.0f;
        int num_scopes = (params.extra_kv != nullptr) ? 2 : 1;
        for (int scope_idx = 0; scope_idx < num_scopes; scope_idx++) {
            const bool is_extra_scope = (scope_idx != 0);
            int scope_topk = is_extra_scope ? params.extra_topk : smem_topk;
            int scope_topk_length = scope_topk;
            if (is_extra_scope) {
                if (params.extra_topk_length != nullptr) {
                    scope_topk_length = __ldg(params.extra_topk_length + batch_idx);
                }
            } else if (params.topk_length != nullptr) {
                scope_topk_length = __ldg(params.topk_length + batch_idx);
            }
            scope_topk_length = min(scope_topk_length, scope_topk);

            int* scope_indices = is_extra_scope ? params.extra_indices : params.indices;
            int scope_stride_indices_b = is_extra_scope ? params.stride_extra_indices_b : params.stride_indices_b;
            int scope_stride_indices_s_q = is_extra_scope ? params.stride_extra_indices_s_q : params.stride_indices_s_q;
            fp8* scope_kv = (fp8*)(is_extra_scope ? params.extra_kv : params.kv);
            int scope_stride_kv_block = is_extra_scope ? params.stride_extra_kv_block : params.stride_kv_block;
            int scope_stride_kv_row = is_extra_scope ? params.stride_extra_kv_row : params.stride_kv_row;
            int scope_page_block_size = is_extra_scope ? params.extra_page_block_size : params.page_block_size;

            int total_blocks = (scope_topk + TOPK_BLOCK_SIZE - 1) / TOPK_BLOCK_SIZE;
            for (int block_idx = 0; block_idx < total_blocks; block_idx++) {
                // is_kv_valid -- SHARED memory so ALL threads can read ALL entries.
                // Dynamic top-k length masks entries past topk_length; those entries may
                // contain arbitrary values and must not participate in QK, softmax, or PV.
                int* gIdx = scope_indices + batch_idx*scope_stride_indices_b
                           + s_q_idx*scope_stride_indices_s_q + block_idx*TOPK_BLOCK_SIZE;
                for (int i = threadIdx.x; i < TOPK_BLOCK_SIZE; i += NUM_THREADS) {
                    int topk_pos = block_idx * TOPK_BLOCK_SIZE + i;
                    int tok = (topk_pos < scope_topk) ? __ldg(gIdx + i) : -1;
                    plan.is_kv_valid[i] = (topk_pos < scope_topk_length) && (tok != -1);
                }
                __threadfence_block(); __syncthreads(); asm volatile("" ::: "memory");

                // PROBE 1: dump indices as seen by kernel
                if (params.debug_buffer && (params.debug_probe_mask & 1) && threadIdx.x == 0 && blockIdx.x == 0 && batch_idx == 0 && block_idx == 0) {
                    for (int i = 0; i < TOPK_BLOCK_SIZE; i++) {
                        int topk_pos = block_idx * TOPK_BLOCK_SIZE + i;
                        int tok = (topk_pos < scope_topk) ? __ldg(gIdx + i) : -1;
                        int valid = (topk_pos < scope_topk_length) && (tok != -1);
                        params.debug_buffer[DebugProbes::IDX_OFFSET + i*2] = (float)tok;
                        params.debug_buffer[DebugProbes::IDX_OFFSET + i*2 + 1] = (float)valid;
                    }
                }
                __threadfence_block(); __syncthreads(); asm volatile("" ::: "memory");

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
                __threadfence_block(); __syncthreads(); asm volatile("" ::: "memory");

                // K dequant from FP8 cache: see sparse_fp8/../splitkv_mla.cuh producer for reference
                {
                    int toff = block_idx * TOPK_BLOCK_SIZE;
                    for (int t = threadIdx.x; t < TOPK_BLOCK_SIZE; t += NUM_THREADS) {
                        int topk_pos = toff + t;
                        int tok = (topk_pos < scope_topk_length)
                            ? __ldg(scope_indices + batch_idx*scope_stride_indices_b
                                    + s_q_idx*scope_stride_indices_s_q + topk_pos)
                            : -1;
                        static constexpr int TSTRIDE = (MODEL_TYPE == ModelType::V32)
                            ? 0 : (HEAD_DIM_NOPE + 2 * HEAD_DIM_ROPE);
                        const int rs = (MODEL_TYPE == ModelType::V32) ? scope_stride_kv_row : TSTRIDE;
                        bf16* row = plan.k.data() + t * p_dim;
                        // PROBE 2e via LSE: write dequant internals to LSE heads 50-57
                        if (threadIdx.x == 0 && blockIdx.x == 0 && t == 0 && params.lse) {
                            float* lse_p = (float*)params.lse + batch_idx*params.stride_lse_b + s_q_idx*params.stride_lse_s_q;
                            lse_p[50] = (float)__ldg(scope_indices + batch_idx*scope_stride_indices_b + s_q_idx*scope_stride_indices_s_q + topk_pos);
                            lse_p[51] = (float)scope_topk;
                            lse_p[52] = (float)scope_topk_length;
                            lse_p[53] = (float)tok;
                        }
                        if (tok != -1) {
                            int blk = tok / scope_page_block_size;
                            int rel = tok % scope_page_block_size;
                            fp8* gK = scope_kv + blk*scope_stride_kv_block + rel*rs;
                            union { float sf_f32[4]; bf16 sf_bf16[8]; } sf_union;
                            if constexpr (MODEL_TYPE == ModelType::V32) {
                                for (int si=0; si<4; si++)
                                    sf_union.sf_f32[si] = __ldg((const float*)(gK + HEAD_DIM_NOPE) + si);
                            } else {
                                uint8_t* bsc = (uint8_t*)(scope_kv + blk*scope_stride_kv_block)
                                             + scope_page_block_size * TSTRIDE;
                                fp8_e8m0* se8 = (fp8_e8m0*)(bsc + rel * NUM_SCALES);
                                for (int si=0; si<NUM_SCALES; si+=2) {
                                    __nv_bfloat162_raw raw = __nv_cvt_e8m0x2_to_bf162raw(*(__nv_fp8x2_storage_t*)(se8+si));
                                    *reinterpret_cast<__nv_bfloat162_raw*>(sf_union.sf_bf16 + si) = raw;
                                }
                            }
                            float* sf = sf_union.sf_f32;  // for V32 access via sf[idx]
                            bf16* sf_b = sf_union.sf_bf16;  // for MODEL1 access via sf_b[idx]

                            static constexpr int N_NOPE = HEAD_DIM_NOPE / 64;
                            bf16* gK_rope = (MODEL_TYPE == ModelType::V32)
                                ? (bf16*)((uint8_t*)gK + HEAD_DIM_NOPE + 4 * sizeof(float))
                                : (bf16*)(gK + HEAD_DIM_NOPE);

                            for (int dt = p_start; dt < p_end; dt++) {
                                int ld = dt - p_start;
                                if (dt < N_NOPE) {
                                    #pragma unroll 1  // don't unroll to reduce register pressure
                                    for (int sub = 0; sub < 4; sub++) {
                                        fp8x16 src;
                                        for (int bi=0; bi<8; bi++) {
                                            ((uint8_t*)&src.lo)[bi] = gK[dt*64 + sub*16 + bi];
                                            ((uint8_t*)&src.hi)[bi] = gK[dt*64 + sub*16 + 8 + bi];
                                        }
                                        bf16 sc = (MODEL_TYPE == ModelType::V32) ? (bf16)sf[dt/2] : sf_b[dt];
                                        bf16x8 lo = cvt_fp8x8_bf16x8(src.lo, __bfloat162bfloat162(*(__nv_bfloat16*)&sc));
                                        bf16x8 hi = cvt_fp8x8_bf16x8(src.hi, __bfloat162bfloat162(*(__nv_bfloat16*)&sc));
                                        // printf debug: first token, first dim tile
                                        if (threadIdx.x == 0 && blockIdx.x == 0 && t == 0 && dt == p_start && sub == 0) {
                                            unsigned long long gK_addr = (unsigned long long)gK;
                                            unsigned long long kv_addr = (unsigned long long)(params.kv);
                                            unsigned long long offset = gK_addr - kv_addr;
                                            printf("DEQUANT: tok=%d blk=%d rel=%d offset=%llu gK[0]=0x%02x lo[0]=%.10f\n",
                                                   tok, blk, rel, offset,
                                                   (int)((uint8_t)gK[0]),
                                                   (float)((bf16*)&lo)[0]);
                                        }
                                        for (int bi = 0; bi < 8; bi++) {
                                            row[ld*64 + sub*16 + bi] = ((bf16*)&lo)[bi];
                                            row[ld*64 + sub*16 + 8 + bi] = ((bf16*)&hi)[bi];
                                        }
                                    }
                                } else {
                                    int rd = dt - N_NOPE;
                                    #pragma unroll 1  // don't unroll to reduce register pressure
                                    for (int sub = 0; sub < 4; sub++) {
                                        bf16x8 lo, hi;
                                        for (int bi = 0; bi < 8; bi++) {
                                            ((bf16*)&lo)[bi] = gK_rope[rd*64 + sub*16 + bi];
                                            ((bf16*)&hi)[bi] = gK_rope[rd*64 + sub*16 + 8 + bi];
                                        }
                                        for (int bi = 0; bi < 8; bi++) {
                                            row[ld*64 + sub*16 + bi] = ((bf16*)&lo)[bi];
                                            row[ld*64 + sub*16 + 8 + bi] = ((bf16*)&hi)[bi];
                                        }
                                    }
                                }
                            }
                        } else {
                            for (int i = 0; i < p_dim; i++) row[i] = bf16(0.0f);
                        }
                    }
                }
                __threadfence_block(); __syncthreads(); asm volatile("" ::: "memory");

                // PROBE 2: dump dequantized K after first pass + raw FP8 bytes for first token
                if (params.debug_buffer && (params.debug_probe_mask & 2) && threadIdx.x == 0 && blockIdx.x == 0 && batch_idx == 0 && block_idx == 0 && pass == 0) {
                    int p_dim0 = p_dim;  // first pass p_dim
                    for (int t = 0; t < TOPK_BLOCK_SIZE; t++) {
                        for (int d = 0; d < p_dim0; d++) {
                            params.debug_buffer[DebugProbes::K_OFFSET + t * 576 + d] = (float)plan.k.data()[t * p_dim0 + d];
                        }
                    }
                }
                // PROBE 2b: dump kernel pointer info + raw FP8 bytes
                if (params.debug_buffer && (params.debug_probe_mask & 2) && threadIdx.x == 0 && blockIdx.x == 0 && batch_idx == 0 && block_idx == 0 && pass == 0) {
                    // Dump pointer info
                    unsigned long long kv_ptr = (unsigned long long)(scope_kv);
                    // Magic number to verify probe data integrity
                    params.debug_buffer[DebugProbes::QK_OFFSET - 1] = 12345.0f;
                    params.debug_buffer[DebugProbes::QK_OFFSET + 0] = (float)(kv_ptr & 0xFFFFFFFFull);
                    params.debug_buffer[DebugProbes::QK_OFFSET + 1] = (float)(kv_ptr >> 32);
                    params.debug_buffer[DebugProbes::QK_OFFSET + 2] = (float)scope_stride_kv_block;
                    params.debug_buffer[DebugProbes::QK_OFFSET + 3] = (float)scope_stride_kv_row;
                    params.debug_buffer[DebugProbes::QK_OFFSET + 4] = (float)scope_page_block_size;

                    // Read the first token's FP8 data
                    int topk_pos0 = block_idx * TOPK_BLOCK_SIZE + 0;
                    int tok0 = (topk_pos0 < scope_topk) ? __ldg(scope_indices + batch_idx*scope_stride_indices_b + s_q_idx*scope_stride_indices_s_q + topk_pos0) : -1;
                    params.debug_buffer[DebugProbes::QK_OFFSET + 5] = (float)tok0;
                    if (tok0 != -1) {
                        int blk0 = tok0 / scope_page_block_size;
                        int rel0 = tok0 % scope_page_block_size;
                        params.debug_buffer[DebugProbes::QK_OFFSET + 6] = (float)blk0;
                        params.debug_buffer[DebugProbes::QK_OFFSET + 7] = (float)rel0;
                        static constexpr int rs_fp8 = (MODEL_TYPE == ModelType::V32) ? 0 : (HEAD_DIM_NOPE + 2 * HEAD_DIM_ROPE);
                        const int rs_actual = (MODEL_TYPE == ModelType::V32) ? scope_stride_kv_row : rs_fp8;
                        unsigned char* gK_raw = (unsigned char*)(scope_kv) + blk0*scope_stride_kv_block + rel0*rs_actual;
                        // Dump first 64 bytes at offset QK_OFFSET+8
                        for (int b = 0; b < 64; b++) {
                            params.debug_buffer[DebugProbes::QK_OFFSET + 8 + b] = (float)(gK_raw[b]);
                        }
                    }
                }

                // PROBE 2c: read bytes from params.kv directly + exact pointer via debug_indices
                if (params.debug_buffer && (params.debug_probe_mask & 2) && threadIdx.x == 0 && blockIdx.x == 0 && batch_idx == 0 && block_idx == 0 && pass == 0) {
                    // Store pointer as 2 ints in debug_indices (lossless!)
                    if (params.debug_indices) {
                        unsigned long long ptr_val = (unsigned long long)(params.kv);
                        params.debug_indices[0] = (int)(ptr_val & 0xFFFFFFFFull);
                        params.debug_indices[1] = (int)(ptr_val >> 32);
                        unsigned long long q_ptr_val = (unsigned long long)(params.q);
                        params.debug_indices[2] = (int)(q_ptr_val & 0xFFFFFFFFull);
                        params.debug_indices[3] = (int)(q_ptr_val >> 32);
                        params.debug_indices[4] = params.topk;
                        params.debug_indices[5] = params.b;
                        params.debug_indices[6] = params.stride_kv_block;
                        params.debug_indices[7] = params.stride_kv_row;
                    }
                    // Read raw bytes from params.kv at offset 0
                    unsigned char* kv_base = (unsigned char*)(params.kv);
                    for (int b = 0; b < 64; b++) {
                        params.debug_buffer[DebugProbes::QK_OFFSET + 72 + b] = (float)(kv_base[b]);
                    }
                }

                // SCALAR QK (debug: bypass MMA to isolate dequant issues)
                {
                    int ar0 = lane_id % 8, ar1 = ar0 + 8;
                    int bn0 = lane_id / 8, bn1 = bn0 + 4;
                    for (int ns = 0; ns < TOPK_BLOCK_SIZE/8; ns++) {
                        int row0 = mm_row + ar0, row1 = mm_row + ar1;
                        int tok0 = ns * 8 + bn0, tok1 = ns * 8 + bn1;
                        float v00=0, v01=0, v10=0, v11=0;
                        for (int k = 0; k < p_dim; k++) {
                            float q0=(float)plan.q.data()[row0*p_dim+k];
                            float q1=(float)plan.q.data()[row1*p_dim+k];
                            float k0=(float)plan.k.data()[tok0*p_dim+k];
                            float k1=(float)plan.k.data()[tok1*p_dim+k];
                            v00+=q0*k0; v01+=q0*k1;
                            v10+=q1*k0; v11+=q1*k1;
                        }
                        int pb=ns*4;
                        rP[pb]+=v00; rP[pb+1]+=v01; rP[pb+2]+=v10; rP[pb+3]+=v11;
                    }
                }
                __threadfence_block(); __syncthreads(); asm volatile("" ::: "memory");
            }  // QK passes

            // ---- Online softmax ----
            // rP layout: ns*4+0/1 = row group 0, ns*4+2/3 = row group 1
            // rO layout: interleaved per MMA tile:
            //   rO[vh*128 + ns*4 + 0] = row0, rO[vh*128 + ns*4 + 1] = row0
            //   rO[vh*128 + ns*4 + 2] = row1, rO[vh*128 + ns*4 + 3] = row1
            {
                float scale_old[2];
                // Phase 1: compute new max and scale_old for both rows
                for (int lr = 0; lr < 2; lr++) {
                    float cm = -INFINITY;
                    int pb_base = lr * 2;
                    for (int ns = 0; ns < TOPK_BLOCK_SIZE/8; ns++) {
                        int pb = ns*4 + pb_base;
                        int col0 = ns*8 + c_col0, col1 = col0 + 4;
                        if (col0 < TOPK_BLOCK_SIZE && plan.is_kv_valid[col0]) cm = fmaxf(cm, rP[pb]);
                        if (col1 < TOPK_BLOCK_SIZE && plan.is_kv_valid[col1]) cm = fmaxf(cm, rP[pb+1]);
                    }
                    cm = fmaxf(cm, __shfl_xor_sync(0xffffffff, cm, 8));
                    cm = fmaxf(cm, __shfl_xor_sync(0xffffffff, cm, 16));
                    cm *= params.sm_scale_div_log2;
                    float om = rM[lr]; rM[lr] = fmaxf(cm, om);
                    scale_old[lr] = exp2f(om - rM[lr]);
                }
                // Phase 2: rescale rO with correct interleaved layout
                for (int vh_i = 0; vh_i < 2; vh_i++) {
                    int vh_base = vh_i * 128;
                    for (int ns = 0; ns < 32; ns++) {
                        int ob = vh_base + ns * 4;
                        rO[ob + 0] *= scale_old[0];
                        rO[ob + 1] *= scale_old[0];
                        rO[ob + 2] *= scale_old[1];
                        rO[ob + 3] *= scale_old[1];
                    }
                }
                // Phase 3: compute exp, sum, update rL
                for (int lr = 0; lr < 2; lr++) {
                    int pb_base = lr * 2;
                    float cs = 0;
                    for (int ns = 0; ns < TOPK_BLOCK_SIZE/8; ns++) {
                        int pb = ns*4 + pb_base;
                        int col0 = ns*8 + c_col0, col1 = col0 + 4;
                        if (col0 < TOPK_BLOCK_SIZE) {
                            rP[pb] = plan.is_kv_valid[col0] ? exp2f(rP[pb]*params.sm_scale_div_log2 - rM[lr]) : 0;
                            rP[pb+1] = plan.is_kv_valid[col1] ? exp2f(rP[pb+1]*params.sm_scale_div_log2 - rM[lr]) : 0;
                            if (plan.is_kv_valid[col0]) cs += rP[pb];
                            if (plan.is_kv_valid[col1]) cs += rP[pb+1];
                        }
                    }
                    cs += __shfl_xor_sync(0xffffffff, cs, 8);
                    cs += __shfl_xor_sync(0xffffffff, cs, 16);
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
            float* sS_ptr = plan.s.data();
            int sr0 = mm_row + (lane_id % 8);
            int sr1 = sr0 + 8;
            for (int ns = 0; ns < TOPK_BLOCK_SIZE/8; ns++) {
                int pb = ns * 4;
                int sc0 = ns * 8 + (lane_id / 8);
                int sc1 = sc0 + 4;
                if (sc0 < TOPK_BLOCK_SIZE) {
                    sS_ptr[sr0 * TOPK_BLOCK_SIZE + sc0] = rP[pb];
                    sS_ptr[sr0 * TOPK_BLOCK_SIZE + sc1] = rP[pb + 1];
                    sS_ptr[sr1 * TOPK_BLOCK_SIZE + sc0] = rP[pb + 2];
                    sS_ptr[sr1 * TOPK_BLOCK_SIZE + sc1] = rP[pb + 3];
                }
            }
            __threadfence_block(); __syncthreads(); asm volatile("" ::: "memory");

            // ---- PV: S x V -> O ----
            for (int vh = 0; vh < 2; vh++) {
                static constexpr int HV = HEAD_DIM_V / 2;
                // Load V from FP8 cache into k buffer
                {
                    int toff = block_idx * TOPK_BLOCK_SIZE;
                    for (int t = threadIdx.x; t < TOPK_BLOCK_SIZE; t += NUM_THREADS) {
                        int topk_pos = toff + t;
                        int tok = (topk_pos < scope_topk_length)
                            ? __ldg(scope_indices + batch_idx*scope_stride_indices_b
                                    + s_q_idx*scope_stride_indices_s_q + topk_pos)
                            : -1;
                        bf16* vrow = plan.k.data() + t * HV;
                        if (tok != -1) {
                            int blk = tok / scope_page_block_size;
                            int rel = tok % scope_page_block_size;
                            static constexpr int TSV = (MODEL_TYPE == ModelType::V32) ? 0 : (HEAD_DIM_NOPE + 2 * HEAD_DIM_ROPE);
                            const int rsv = (MODEL_TYPE == ModelType::V32) ? scope_stride_kv_row : TSV;
                            fp8* gK = scope_kv + blk*scope_stride_kv_block + rel*rsv;
                            union { float sf_f32[4]; bf16 sf_bf16[8]; } sf_union_v;
                            if constexpr (MODEL_TYPE==ModelType::V32)
                                for (int si=0; si<4; si++) sf_union_v.sf_f32[si] = ((const float*)(gK + HEAD_DIM_NOPE))[si];
                            else {
                                static constexpr int TS2 = HEAD_DIM_NOPE + 2 * HEAD_DIM_ROPE;
                                uint8_t* bsc2 = (uint8_t*)(scope_kv + blk*scope_stride_kv_block)
                                               + scope_page_block_size * TS2;
                                fp8_e8m0* se2 = (fp8_e8m0*)(bsc2 + rel * NUM_SCALES);
                                for (int si=0; si<NUM_SCALES; si+=2) {
                                    __nv_bfloat162_raw raw = __nv_cvt_e8m0x2_to_bf162raw(*(__nv_fp8x2_storage_t*)(se2+si));
                                    *reinterpret_cast<__nv_bfloat162_raw*>(sf_union_v.sf_bf16 + si) = raw;
                                }
                            }
                            float* sf_v = sf_union_v.sf_f32;
                            bf16* sf_vb = sf_union_v.sf_bf16;
                            int vo = vh*HV;
                            for (int vi = 0; vi < HV/16; vi++) {
                                int vd = vo + vi*16;
                                if constexpr (MODEL_TYPE == ModelType::MODEL1) {
                                    // Dims 448-511 overlap with K RoPE (bf16) region; load directly as bf16
                                    if (vd >= HEAD_DIM_NOPE) {
                                        bf16x8 lo, hi;
                                        bf16* gV_bf16 = (bf16*)(gK + HEAD_DIM_NOPE);
                                        for (int bi = 0; bi < 8; bi++) {
                                            ((bf16*)&lo)[bi] = gV_bf16[(vd - HEAD_DIM_NOPE) + bi];
                                            ((bf16*)&hi)[bi] = gV_bf16[(vd - HEAD_DIM_NOPE) + 8 + bi];
                                        }
                                        for (int bi = 0; bi < 8; bi++) {
                                            vrow[vi*16 + bi] = ((bf16*)&lo)[bi];
                                            vrow[vi*16 + 8 + bi] = ((bf16*)&hi)[bi];
                                        }
                                        continue;
                                    }
                                }
                                fp8x16 src;
                                *reinterpret_cast<uint2*>(&src.lo) = *reinterpret_cast<const uint2*>(gK + vd);
                                *reinterpret_cast<uint2*>(&src.hi) = *reinterpret_cast<const uint2*>(gK + vd + 8);
                                bf16 sc = (vd/QUANT_TILE_SIZE < NUM_SCALES) ? ((MODEL_TYPE==ModelType::V32) ? (bf16)sf_v[vd/QUANT_TILE_SIZE] : sf_vb[vd/QUANT_TILE_SIZE]) : (bf16)1.0f;
                                bf16x8 lo = cvt_fp8x8_bf16x8(src.lo, __bfloat162bfloat162(*(__nv_bfloat16*)&sc));
                                bf16x8 hi = cvt_fp8x8_bf16x8(src.hi, __bfloat162bfloat162(*(__nv_bfloat16*)&sc));
                                for (int bi=0; bi<8; bi++) {
                                    vrow[vi*16 + bi] = ((bf16*)&lo)[bi];
                                    vrow[vi*16 + 8 + bi] = ((bf16*)&hi)[bi];
                                }
                            }
                        } else {
                            for (int i = 0; i < HV; i++) vrow[i] = bf16(0.0f);
                        }
                    }
                }
                __threadfence_block(); __syncthreads(); asm volatile("" ::: "memory");

                // PV MMA via isolated __noinline__ function (no params visibility)
                pv_mma_kernel(plan.s.data(), plan.k.data(), rO,
                              HV, TOPK_BLOCK_SIZE, vh, lane_id, mm_row);
                __threadfence_block(); __syncthreads(); asm volatile("" ::: "memory");
            }
            }  // K/V blocks
        }  // main/extra KV scopes

        // ---- Normalize output, write LSE and O ----
        int row0 = mm_row + (lane_id % 8);
        int row1 = row0 + 8;
        if (row0 < BLOCK_M) { plan.sM[row0] = rM[0]; plan.sL[row0] = rL[0]; }
        if (row1 < BLOCK_M) { plan.sM[row1] = rM[1]; plan.sL[row1] = rL[1]; }
        __threadfence_block(); __syncthreads(); asm volatile("" ::: "memory");

        int max_row = min(params.h_q - start_head_idx, BLOCK_M);
        float o_scale[2];
        bool has_sink = (params.attn_sink != nullptr);
        float attn_sink_val[2] = {0.0f, 0.0f};
        if (has_sink) {
            if (row0 < max_row) attn_sink_val[0] = __ldg((const float*)params.attn_sink + start_head_idx + row0) * (float)M_LOG2E;
            if (row1 < max_row) attn_sink_val[1] = __ldg((const float*)params.attn_sink + start_head_idx + row1) * (float)M_LOG2E;
        }
        for (int lr = 0; lr < 2; lr++) {
            int r = lr == 0 ? row0 : row1;
            float L = plan.sL[r], M = plan.sM[r];
            float denom = L;
            if (has_sink) {
                denom += exp2f(attn_sink_val[lr] - M);
            }
            o_scale[lr] = (denom == 0.0f) ? 0.0f : (1.0f / denom);
        }
        // Apply o_scale with correct interleaved rO layout:
        // rO[vh*128 + ns*4 + 0..1] = row0, rO[vh*128 + ns*4 + 2..3] = row1
        for (int vh_i = 0; vh_i < 2; vh_i++) {
            int vh_base = vh_i * 128;
            for (int ns = 0; ns < 32; ns++) {
                int ob = vh_base + ns * 4;
                rO[ob + 0] *= o_scale[0];
                rO[ob + 1] *= o_scale[0];
                rO[ob + 2] *= o_scale[1];
                rO[ob + 3] *= o_scale[1];
            }
        }

        // Write LSE
        float* gLSE = (float*)params.lse + batch_idx*params.stride_lse_b
                    + s_q_idx*params.stride_lse_s_q + start_head_idx;
        if (row0 < max_row) {
            float L0 = plan.sL[row0], M0 = plan.sM[row0];
            gLSE[row0] = (L0 == 0.0f) ? INFINITY : (logf(L0) + M0 / (float)M_LOG2E);
        }
        if (row1 < max_row) {
            float L1 = plan.sL[row1], M1 = plan.sM[row1];
            gLSE[row1] = (L1 == 0.0f) ? INFINITY : (logf(L1) + M1 / (float)M_LOG2E);
        }
        __threadfence_block(); __syncthreads(); asm volatile("" ::: "memory");

        // Store output
        bf16* gO = (bf16*)params.out + batch_idx*params.stride_o_b
                 + s_q_idx*params.stride_o_s_q + start_head_idx*params.stride_o_h_q;
        for (int ns = 0; ns < HEAD_DIM_V/8; ns++) {
            int ob = ns * 4;
            int oc0 = ns * 8 + (lane_id / 8);
            int oc1 = oc0 + 4;
            if (oc0 < HEAD_DIM_V) {
                if (row0 < max_row) {
                    gO[row0 * params.stride_o_h_q + oc0] = (bf16)rO[ob];
                    gO[row0 * params.stride_o_h_q + oc1] = (bf16)rO[ob + 1];
                }
                if (row1 < max_row) {
                    gO[row1 * params.stride_o_h_q + oc0] = (bf16)rO[ob + 2];
                    gO[row1 * params.stride_o_h_q + oc1] = (bf16)rO[ob + 3];
                }
            }
        }
        __threadfence_block(); __syncthreads(); asm volatile("" ::: "memory");
    }
#endif
}  // devfunc

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

