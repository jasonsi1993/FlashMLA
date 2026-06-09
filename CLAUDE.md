# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project requires CUDA 12.8+, PyTorch 2.0+, and an SM90/SM100/SM120 GPU.

On the local development machine:
- Python venv at `/data/Home/cdsi/pytorch/myenv/bin/python` (Python 3.10, PyTorch 2.9.0a0+cu128)
- GPU: SM120 (Blackwell, compute capability 12.0)
- CUDA: 12.8 (requires `FLASH_MLA_DISABLE_SM100=1` since SM100 needs NVCC 12.9+)
- `kernelkit` is a local package at `tests/kernelkit/` — tests must run from the project root with `PYTHONPATH="tests:$PYTHONPATH"`

## Build

```bash
# Full build (SM120 only — SM100 disabled due to NVCC 12.8)
FLASH_MLA_DISABLE_SM100=1 python setup.py build_ext --inplace

# Rebuild only changed files (touching kernel.cuh forces SM120 recompile)
touch csrc/sm120/decode/sparse_fp8/kernel.cuh && FLASH_MLA_DISABLE_SM100=1 python setup.py build_ext --inplace

# Check register count and spill (critical for correctness):
FLASH_MLA_DISABLE_SM100=1 python setup.py build_ext --inplace 2>&1 | grep -E "Used [0-9]+ registers|spill"
```

## Run Tests

```bash
# Full test suite (from project root):
PYTHONPATH="tests:$PYTHONPATH" python tests/test_flash_mla_sparse_decoding.py

# Single test case via Python:
PYTHONPATH="tests:$PYTHONPATH" python -c "
import torch, flash_mla
torch.set_default_dtype(torch.bfloat16); torch.set_default_device('cuda')
from lib import TestParam, ExtraTestParamForDecode, generate_testcase_for_decode, run_flash_mla_decode
t = TestParam(s_q=1, s_kv=512, topk=64, h_q=64, d_qk=576, d_v=512, seed=0,
    check_correctness=True, num_runs=0, have_attn_sink=False,
    decode=ExtraTestParamForDecode(b=1, is_varlen=False, have_zero_seqlen_k=False, block_size=64))
tcase = generate_testcase_for_decode(t)
ts, ns = flash_mla.get_mla_metadata()
out, lse = run_flash_mla_decode(t, tcase, ts, ns)
torch.cuda.synchronize()
print(out.isnan().sum().item())
"

# Memory error check:
compute-sanitizer --tool memcheck python tests/test_flash_mla_sparse_decoding.py
```

## Architecture

### Dispatch Layer

`csrc/api/sparse_decode.h` is the sparse decoding entry point. It detects GPU architecture at runtime via `Arch arch = Arch()` and routes to the appropriate kernel implementation:

```
Arch Detection:
  is_sm120f() → sm120::decode::sparse_fp8::run_sm120_sparse_decode_kernel (standalone SM80-MMA)
  is_sm100f() → Decode_Sm100_Head64_Impl / Head64x2_Impl / Head128_Impl
  is_sm90a()  → Decode_Sm90_Impl (GMMA-based, shared with SM120 mode)
```

The SM120 dispatch is **hardcoded** to `<ModelType::V32, 64>` — only V3.2 format (d_qk=576) with 64 query heads is supported. MODEL1 (d_qk=512) and h_q=128 are NOT supported on SM120 via this kernel.

### Kernel Implementations

**SM90 Kernel** (`csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`):
- Uses GMMA (warpgroup MMA) via CuTe abstractions
- 384 threads (3 warpgroups: 2 compute + 1 producer)
- TMA for Q loading, cluster launch
- Has `FLASH_MLA_SM120_MODE` for reduced shared memory (~73KB, 2-pass QK)
- BUT: GMMA fences (`wgmma.fence`) are gated by `CUTE_ARCH_MMA_SM90A_ENABLED` which is only defined for `__CUDA_ARCH__ == 900` — NOT available on SM120 hardware

**SM120 Kernel** (`csrc/sm120/decode/sparse_fp8/kernel.cuh`):
- Uses SM80-style `mma.sync.aligned.m16n8k16` (not warpgroup MMA)
- 128 threads (4 warps), single warpgroup
- Manual shared memory loads (cooperative), no TMA
- 2-pass QK (5 tiles + 4 tiles for V32) to fit in ~90KB shared memory
- No split-KV (forces no-split mode)

### Shared Memory Layout (SM120 Kernel)

```
SharedMemoryPlan (~90KB):
  union { {q(40KB), k(40KB)}, oBuf(64KB) }  // 80KB
  s[64][64] bf16         // 8KB  (S = attention weights)
  is_kv_valid[64] bool   // 64B  (shared — all threads read all entries)
  sM[64], sL[64], sScale[64], sOScale[64] float  // 1KB
```

**Critical: Layout conventions**
- Q: `data[row * p_dim + col]` — row-major, stride = p_dim (320 or 256)
- K: `data[tok * p_dim + dim]` — row-major, stride = p_dim (320 or 256)
- V half: `data[tok * HV + dim]` — row-major, stride = HV (256)
- S: `data[row * TOPK_BLOCK_SIZE + col]` — row-major, stride = TOPK_BLOCK_SIZE (64)
- K and V reuse the same `plan.k.data()` buffer (sequentially, separated by `__syncthreads()`)

### FP8 KV Cache Format (V32, d_qk=576)

Each token = 656 bytes:
```
Bytes 0-511:   512 × fp8_e4m3  (NoPE dimensions, quantized)
Bytes 512-527: 4 × float32     (scale factors, one per 128 NoPE dims)
Bytes 528-655: 64 × bf16       (RoPE dimensions, NOT quantized)
```

The RoPE region starts at byte offset 528 from the token base. Scale factors map as: sf[0]→dims 0-127, sf[1]→128-255, sf[2]→256-383, sf[3]→384-511.

### MMA Register Mapping (SM120 Kernel)

The kernel uses CuTe-derived MMA register mapping for `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`:

```
Per thread (128 threads, 4 warps × 32 lanes):
  ar0 = lane_id % 8          → M-row 0 (local, 0-7)
  ar1 = ar0 + 8              → M-row 1 (local, 8-15)
  a_col = (lane_id / 8) * 4  → K-column within 16-wide tile
  bk0 = lane_id % 8          → K-dim position 0
  bk1 = bk0 + 8              → K-dim position 1
  bn0 = lane_id / 8          → N-column (token) position 0
  bn1 = bn0 + 4              → N-column (token) position 1

Per warp (32 lanes):
  mm_row = warp_id * 16      → global M-row offset
  Warp 0: rows 0-15, Warp 1: rows 16-31, Warp 2: rows 32-47, Warp 3: rows 48-63
```

### Softmax Reduction (Critical)

Threads that share the **same row** are separated by 8 in lane_id (e.g., lanes 0, 8, 16, 24 all handle row 0). The max and sum across column groups MUST use shuffle masks `{8, 16}` NOT `{1, 2}`:

```cpp
cm = fmaxf(cm, __shfl_xor_sync(0xffffffff, cm, 8));   // pair lanes (i, i^8)
cm = fmaxf(cm, __shfl_xor_sync(0xffffffff, cm, 16));  // quad lanes (i, i^16)
cs += __shfl_xor_sync(0xffffffff, cs, 8);
cs += __shfl_xor_sync(0xffffffff, cs, 16);
```

Shuffle {1,2} would combine threads 0,1,2,3 which handle DIFFERENT rows — producing wrong results.

### Known Bugs Fixed (SM120 Kernel)

1. **kv_valid thread-local → shared memory** (root cause of NaN): `bool kv_valid[64]` was a per-thread stack array where each thread initialized only ONE entry but the softmax read ALL 64. Fixed by using `plan.is_kv_valid[i]` (shared memory).

2. **RoPE tile loaded as fp8**: The K dequant loop loaded ALL tiles (including RoPE at dt=8) through fp8 dequant. RoPE is stored as bf16 at a different byte offset. Fixed by branching on `dt < HEAD_DIM_NOPE/64`.

3. **K/V shared memory stride mismatch**: Store used `data[t + dim * TOPK_BLOCK_SIZE]` but load used `data[t * TOPK_BLOCK_SIZE + dim]`. Fixed by using `p_dim`/`HV` as consistent stride.

4. **Missing output normalization**: rO was stored without dividing by rL. Added `rO *= o_scale` with `o_scale = 1/rL`.

5. **Missing LSE output**: LSE tensor was uninitialized. Added `logf(L) + M / M_LOG2E` write.

6. **Missing attn_sink**: All test cases set `have_attn_sink=True`. Added attn_sink read and `1/(rL + exp2(attn_sink - rM))` normalization.

## Key Files

| File | Role |
|------|------|
| `csrc/api/sparse_decode.h` | Dispatch logic, architecture detection, parameter setup |
| `csrc/sm120/decode/sparse_fp8/kernel.cuh` | SM120 sparse decode kernel (actively modified) |
| `csrc/sm120/decode/sparse_fp8/config.h` | SM120 kernel constants, SharedMemoryPlan, TiledMMA def |
| `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh` | SM90 GMMA kernel (reference for correct behavior) |
| `csrc/sm90/decode/sparse_fp8/config.h` | SM90 kernel config with `FLASH_MLA_SM120_MODE` |
| `tests/lib.py` | Test data generation, kernel invocation helpers |
| `tests/ref.py` | CPU reference implementations |
| `tests/quant.py` | FP8 quantization/dequantization (KV cache format) |
| `tests/kernelkit/` | Test utilities (bench, compare, precision checks) |
| `flash_mla/flash_mla_interface.py` | Python API |
| `setup.py` | Build configuration, NVCC flags, architecture targets |

## Build Flags

- `FLASH_MLA_DISABLE_SM100=1` — skip SM100 compilation (required for NVCC < 12.9)
- `FLASH_MLA_DISABLE_SM120=1` — skip SM120 compilation
- `FLASH_MLA_DISABLE_SM90=1` — skip SM90 compilation
- `FLASH_MLA_HOST_HAS_SM120` — enables `FLASH_MLA_SM120_MODE` in SM90 kernel config
- `--use_fast_math` — NVCC flag applied globally; NOT the cause of NaN issues (verified)
