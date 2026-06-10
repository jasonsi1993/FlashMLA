#pragma once

#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

using namespace cute;

namespace sm120::decode::sparse_fp8 {

template<ModelType MODEL_TYPE, int NUM_HEADS>
class KernelTemplate {
public:

static_assert(NUM_HEADS == 64 || NUM_HEADS == 128);

static constexpr int HEAD_DIM_K = MODEL_TYPE == ModelType::V32 ? 576 : 512;
static constexpr int HEAD_DIM_V = 512;
static constexpr int HEAD_DIM_ROPE = 64;
static constexpr int HEAD_DIM_NOPE = HEAD_DIM_K - HEAD_DIM_ROPE;

static constexpr int QUANT_TILE_SIZE = MODEL_TYPE == ModelType::V32 ? 128 : 64;
static constexpr int NUM_SCALES = MODEL_TYPE == ModelType::V32 ? 4 : 8;

// SM80 MMA tile: 16x8x16 (M×N×K)
static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 16;

static constexpr int NUM_THREADS = 128;  // 4 warps, simpler than GMMA's 384
static constexpr int BLOCK_M = 64;
static constexpr int TOPK_BLOCK_SIZE = 64;

// Use pass splitting to fit in 99KB shared memory
// 5 tiles = 320 dims per pass → 2 passes for both V32(9 tiles) and MODEL1(8 tiles)
static constexpr int NUM_K_BUFS = 1;
static constexpr int QK_TILES_PER_PASS = 5;
static constexpr int NUM_QK_PASSES = (HEAD_DIM_K/64 + QK_TILES_PER_PASS - 1) / QK_TILES_PER_PASS;

// SM80 MMA atom for BF16: 16×8×16 per atom, per-warp tiling
// Each warp processes 16 rows independently; 4 warps cover all 64 heads
using MMA_Atom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
// QK: 16 rows × 64 columns per warp (1 M-atom × 8 N-atoms, K=16)
using TiledMMA_QK = decltype(make_tiled_mma(
    MMA_Atom{},
    Layout<Shape<_1, _8>>{},   // 1 M-atom × 8 N-atoms = 8 atoms per warp
    Tile<_16, _64, _16>{}));   // 16×64 output per warp, K=16
// PV: same MMA atom, used for S×V with 64-column V tiles
using TiledMMA_PV = decltype(make_tiled_mma(
    MMA_Atom{},
    Layout<Shape<_1, _8>>{},   // 1 M-atom × 8 N-atoms per warp
    Tile<_16, _64, _16>{}));   // same tiling, reused for V-column iteration

// Simple row-major shared memory layouts (no GMMA atoms needed)
using SmemLayoutQPass = Layout<Shape<Int<BLOCK_M>, Int<QK_TILES_PER_PASS*64>>, Stride<_1, Int<BLOCK_M>>>;
using SmemLayoutKPass = Layout<Shape<Int<TOPK_BLOCK_SIZE>, Int<QK_TILES_PER_PASS*64>>, Stride<_1, Int<TOPK_BLOCK_SIZE>>>;
using SmemLayoutOBuf = Layout<Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>, Stride<_1, Int<BLOCK_M>>>;
using SmemLayoutS = Layout<Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>, Stride<_1, Int<BLOCK_M>>>;

// V layouts for PV computation
using SmemLayoutHalfV = Layout<Shape<Int<TOPK_BLOCK_SIZE>, Int<HEAD_DIM_V/2>>, Stride<_1, Int<TOPK_BLOCK_SIZE>>>;
using SmemLayoutV = Layout<Shape<Int<TOPK_BLOCK_SIZE>, Int<HEAD_DIM_V>>, Stride<_1, Int<TOPK_BLOCK_SIZE>>>;

struct SharedMemoryPlan {
    // q and k in union with oBuf; forced no-split mode
    union {
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutQPass>> q;  // ~40KB
            array_aligned<bf16, cosize_v<SmemLayoutKPass>> k;  // ~40KB
        };
        array_aligned<bf16, cosize_v<SmemLayoutOBuf>> oBuf;    // 64KB
    };
    CUTE_ALIGNAS(1024) array_aligned<bf16, cosize_v<SmemLayoutS>> s;  // 8KB
    bool is_kv_valid[TOPK_BLOCK_SIZE];

    float sM[BLOCK_M], sL[BLOCK_M], sScale[BLOCK_M], sOScale[BLOCK_M];
};

template<typename Shape_Q, typename TMA_Q>
struct TmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
    CUtensorMap tensor_map_o;
};

// Synchronization
static __forceinline__ __device__ void sync_all_threads() {
    __syncthreads();
}

template<typename TMAParams>
static __device__ void devfunc(
    const SparseAttnDecodeParams &params,
    const TMAParams &tma_params);

static void run(const SparseAttnDecodeParams &params);

};

}  // namespace sm120::decode::sparse_fp8
