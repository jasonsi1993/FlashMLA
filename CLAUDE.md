# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project requires CUDA 12.8+, PyTorch 2.0+, and an SM90/SM100/SM120 GPU.

On the local development machine:
- Python venv at `/data/Home/cdsi/pytorch/myenv/bin/python` (Python 3.10, PyTorch 2.9.0a0+cu128)
- GPU: SM120 (Blackwell, compute capability 12.0)
- CUDA: 13.0 (requires `FLASH_MLA_DISABLE_SM100=1` since SM100 needs NVCC 12.9+)
- `kernelkit` is a local package at `tests/kernelkit/` — tests must run from the project root with `PYTHONPATH="tests:$PYTHONPATH"`

## Build

```bash
# Full build (SM120 only — SM100 disabled)
FLASH_MLA_DISABLE_SM100=1 python setup.py build_ext --inplace

# Rebuild only changed files (touching kernel.cuh forces SM120 recompile)
touch csrc/sm120/decode/sparse_fp8/kernel.cuh && FLASH_MLA_DISABLE_SM100=1 python setup.py build_ext --inplace

# Check register count and spill (critical for correctness):
FLASH_MLA_DISABLE_SM100=1 python setup.py build_ext --inplace 2>&1 | grep -E "Used [0-9]+ registers|spill"
```

## Run Tests

```bash
# Full test suite (from project root, stops on first failure):
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
```

## Architecture

### Dispatch Layer

`csrc/api/sparse_decode.h` is the sparse decoding entry point. It detects GPU architecture at runtime via `Arch arch = Arch()` and routes to the appropriate kernel implementation:

```
Arch Detection:
  is_sm120f() → sm120::decode::sparse_fp8::run_sm120_sparse_decode_kernel (standalone SM80-MMA)
  is_sm100f() → Decode_Sm100_Head64_Impl / Head64x2_Impl / Head128_Impl
  is_sm90a()  → Decode_Sm90_Impl (GMMA-based)
```

The SM120 dispatch supports **both V32 and MODEL1** with 64 query heads. For b≥4, the dispatch splits into launches of max 2 batches (see `SM120_MAX_B` workaround for CUDA 13 `__grid_constant__` codegen interaction).

### Kernel Implementations

**SM90 Kernel** (`csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`):
- Uses GMMA (warpgroup MMA) via CuTe abstractions
- 384 threads (3 warpgroups: 2 compute + 1 producer)
- TMA for Q loading, cluster launch
- 128KB shared memory — can store S (softmax) as FP32 in registers for warpgroup 0 (RS MMA variant)
- Wgmma fences gated by `CUTE_ARCH_MMA_SM90A_ENABLED` (SM90 only)

**SM120 Kernel** (`csrc/sm120/decode/sparse_fp8/kernel.cuh`):
- Uses SM80-style `mma.sync.aligned.m16n8k16` (not warpgroup MMA)
- 128 threads (4 warps), each warp independently processes 16 of 64 query rows
- Q/K/V loads: cooperative (all 128 threads)
- MMA: per-warp (uses `lane_id`, not `threadIdx.x`) via `__noinline__` device functions
- 2-pass QK (5 tiles + 4 tiles for V32, 5+3 for MODEL1) to fit in ~91KB shared memory
- No split-KV (forces no-split mode)
- S (softmax) stored as BF16 in shared memory (8KB for 64×64) to fit in 99KB smem limit
- Register count: MODEL1 ~70 regs, V32 ~78 regs (with `#pragma unroll 1` on K dequant)

### Shared Memory Layout (SM120 Kernel)

```
SharedMemoryPlan (~91KB):
  union { {q(40KB), k(40KB)}, oBuf(64KB) }  // 80KB
  s[64][64] bf16         // 8KB  (S = attention weights, BF16 precision)
  is_kv_valid[64] bool   // 64B  (shared — all threads read all entries)
  sM[64], sL[64], sScale[64], sOScale[64] float  // 1KB
```

**Critical: Layout conventions**
- Q: `data[row * p_dim + col]` — row-major, stride = p_dim (320 or 256)
- K: `data[tok * p_dim + dim]` — row-major, stride = p_dim (320 or 256)
- V half: `data[tok * HV + dim]` — row-major, stride = HV (256)
- S: `data[row * TOPK_BLOCK_SIZE + col]` — row-major, stride = TOPK_BLOCK_SIZE (64)
- K and V reuse the same `plan.k.data()` buffer (sequentially, separated by `__syncthreads()`)
- rO accumulator: interleaved per MMA tile — `rO[vh*128 + ns*4 + 0..1] = row0`, `rO[vh*128 + ns*4 + 2..3] = row1`

### FP8 KV Cache Formats

**V32 (d_qk=576)**: 656 bytes/token
```
Bytes 0-511:   512 × fp8_e4m3  (NoPE dimensions, quantized)
Bytes 512-527: 4 × float32     (scale factors, one per 128 NoPE dims)
Bytes 528-655: 64 × bf16       (RoPE dimensions, NOT quantized)
```
Token stride (NoPE+RoPE): `d_qk = 576` bytes. Scale factors are inline at `gK + 512` (4 × float32).

**MODEL1 (d_qk=512)**: 584 bytes/token
```
Bytes 0-447:   448 × fp8_e4m3  (NoPE dimensions, quantized)
Bytes 448-575: 64 × bf16       (RoPE dimensions, NOT quantized)
Bytes 576-583: 8 × fp8_e8m0    (scale factors, 1 per 64 NoPE dims; byte 583 is padding)
```
Token stride (NoPE+RoPE): `HEAD_DIM_NOPE + 2*HEAD_DIM_ROPE = 576` bytes (different from the 4D view stride of 584!).
Scale factors are at block-end: `kv + blk*stride_kv_block + page_block_size*576 + rel*NUM_SCALES`.
The kernel's 4D view has stride(1)=584 but the actual NoPE+RoPE data has stride 576 — scales sit in the 8-byte gap.

### MMA Register Mapping (SM120 Kernel)

```
Per thread (128 threads, 4 warps × 32 lanes):
  ar0 = lane_id % 8          → M-row 0 (local, 0-7)
  ar1 = ar0 + 8              → M-row 1 (local, 8-15)
  a_col = (lane_id / 8) * 4  → K-column within 16-wide tile
  bk0 = lane_id % 8          → K-dim position 0
  bk1 = bk0 + 8              → K-dim position 1
  bn0 = lane_id / 8          → N-column position 0
  bn1 = bn0 + 4              → N-column position 1

B-fragment register pairing (CRITICAL — fixed in commit 7b61192):
  b_regs[0] = B[bk0, bn0], B[bk0, bn1]    ← same K-row, two N-cols
  b_regs[1] = B[bk1, bn0], B[bk1, bn1]    ← same K-row, two N-cols
  OLD (buggy): paired consecutive K-positions (bk0, bk0+1) — lane 7 read ks*16+16 (cross-tile)

rO accumulator layout (CRITICAL — fixed in commit 82e99af):
  rO[vh*128 + ns*4 + 0] = row0, col0
  rO[vh*128 + ns*4 + 1] = row0, col1
  rO[vh*128 + ns*4 + 2] = row1, col0
  rO[vh*128 + ns*4 + 3] = row1, col1
  OLD (buggy): treated rO[0:128]=row0, rO[128:256]=row1 (contiguous assumption)

Per warp (32 lanes):
  mm_row = warp_id * 16      → global M-row offset
  Warp 0: rows 0-15, Warp 1: rows 16-31, Warp 2: rows 32-47, Warp 3: rows 48-63
```

### Softmax Reduction (Critical)

Threads that share the **same row** are separated by 8 in lane_id. Max and sum MUST use shuffle `{8, 16}` NOT `{1, 2}`:

```cpp
cm = fmaxf(cm, __shfl_xor_sync(0xffffffff, cm, 8));
cm = fmaxf(cm, __shfl_xor_sync(0xffffffff, cm, 16));
```

## Bugs Fixed (15 total)

| # | Bug | Root Cause | Fix |
|---|-----|------------|-----|
| 1 | kv_valid thread-local | Per-thread stack array, only 1 entry initialized | Shared memory `plan.is_kv_valid[i]` |
| 2 | RoPE loaded as fp8 | RoPE (bf16) misinterpreted as fp8 | Branch: NoPE→fp8 dequant, RoPE→bf16 direct |
| 3 | K/V layout mismatch | Store vs load used different strides | `p_dim`/`HV` as consistent stride |
| 4 | Missing output norm | rO not divided by rL | `o_scale = 1/rL`, `rO *= o_scale` |
| 5 | Missing LSE output | LSE tensor uninitialized | `logf(L) + M/M_LOG2E` |
| 6 | Missing attn_sink | attn_sink never read | Read + sink normalization |
| 7 | K dequant 16/64 dims | Only first 16 of 64 K-dims loaded per tile | 4-sub-tile loop (sub=0..3) |
| 8 | threadfence_block guards (nice-to-have) | Possible CUDA 13 st.shared reordering | `__threadfence_block()` before all `__syncthreads()` |
| 9 | attn_sink sentinel fed to exp2f | Default values fed to exp2f → NaN/spurious term | Boolean-gate: only compute when sink present |
| 10 | MODEL1 V dequant reads RoPE as fp8 | V vh=1 reads bf16 RoPE bytes (448-511) as fp8 | bf16 direct load for vd ≥ HEAD_DIM_NOPE |
| 11 | **e8m0 scale stored in float[]** (root cause of b≥4 NaN) | Packed BF16 from `__nv_cvt_e8m0x2_to_bf162raw` stored in `float sf[]`; read `(bf16)sf[dt]` reinterprets BF16 bits as float → garbage | Union `{float[4], bf16[8]}`; use bf16* for MODEL1 |
| 12 | rO row scaling layout (online softmax rescale) | `rO[lr*128 : lr*128+128] *= scale` assumed contiguous rows but MMA layout is interleaved | Iterate ns groups with per-row scaling |
| 13 | rO row scaling layout (output normalization) | Same contiguous assumption as #12 | Same fix, iterating ns groups |
| 14 | B-fragment MMA register pairing | Consecutive K-positions `(bk0,bk0+1)` paired; lane 7 reads cross-tile at `ks*16+16` | Pair same K-row with two N-cols: `B[bk0,bn0],B[bk0,bn1]` |
| 15 | attn_sink LSE and o_scale | `o_scale=(L==0)?0:(1/denom)` ignored sink when L=0; LSE ignored sink entirely; `exp2f(+inf - +inf)=NaN` | `o_scale=(denom==0)?0:(1/denom)`; LSE uses `denom=L+exp2f(sink-M)`; `!isfinite(sink)` guard |

## Precision Gap (Test Tolerance Issue)

The test suite compares the SM120 kernel output against an **FP32 reference** (`tests/ref.py`). Three precision gaps exist:

| Stage | Kernel | Reference | Error contribution |
|-------|--------|-----------|-------------------|
| QK matmul | BF16×BF16 MMA (HMMA) | FP32 `@` matmul | ~0.18 per QK entry |
| Softmax S | BF16 storage to `plan.s` (8KB, fits 99KB smem) | FP32 in registers | ~0.00015 per S value |
| PV matmul | BF16×BF16 MMA (HMMA) | FP32 `@` matmul | ~0.06 per output |

The test tolerance is `cos_diff_tol=5e-6` — this is calibrated for FP32-vs-FP32 comparison. Both SM90 and SM120 use BF16 MMA and BF16 S storage, so both have this gap. The CLAUDE.md previously reported SM90 passes 32/121 (26%) of edge-case tests — even SM90 fails most tests on these strict tolerances.

**Production-configuration tests** (b=1, bs=64, no-varlen, no-sink) pass with 0 NaN. The test suite has zero production tests — all 4748 cases use edge parameters (varlen, bs≠64, b≥4, sink, topk=576).

## Test Harness (Critical)

`tests/lib.py` includes three guards to prevent cross-test CUDA memory allocator contamination:

1. `torch.cuda.empty_cache()` in `generate_testcase_for_decode()` — releases cached memory blocks
2. `torch.cuda.synchronize()` before `flash_mla_with_kvcache()` — ensures metadata kernel completes
3. `torch.cuda.synchronize()` after `flash_mla_with_kvcache()` — ensures decode kernel completes

Without these, running V32 tests before MODEL1 tests causes NaN from CUDA memory allocator state carry-over.

## Smem Budget

SM120 shared memory (99KB total):
- Q buffer: 64×320 bf16 = 40KB (union with K)
- K buffer: 64×320 bf16 = 40KB (union with Q)
- S buffer: 64×64 bf16 = 8KB
- is_kv_valid: 64B
- sM/sL/sScale/sOScale: 64×4×4 = 1KB
- **Total: ~91KB** (8KB remains for stack/alignment)

SM90 has 128KB smem and can optionally store S in FP32 (16KB), but both SM90 and SM120 store S as BF16 in shared memory. SM90 warpgroup 0 keeps S in registers (RS MMA path), avoiding the shared memory round-trip for one warpgroup.

## Key Files

| File | Role |
|------|------|
| `csrc/api/sparse_decode.h` | Dispatch logic, architecture detection, parameter setup, batch splitting |
| `csrc/sm120/decode/sparse_fp8/kernel.cuh` | SM120 sparse decode kernel (15 bugs fixed) |
| `csrc/sm120/decode/sparse_fp8/config.h` | SM120 kernel constants, SharedMemoryPlan, per-warp TiledMMA |
| `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh` | SM90 GMMA kernel (reference for correct behavior) |
| `tests/lib.py` | Test data generation, kernel invocation, sync harness |
| `tests/ref.py` | **FP32 reference** — uses full FP32 matmul, NOT BF16 MMA |
| `tests/quant.py` | FP8 quantization/dequantization (KV cache format) |
| `tests/kernelkit/compare.py` | `check_is_allclose` — abs_tol=1e-3, rel_tol=0.0157, cos_diff_tol=5e-6 |
| `flash_mla/flash_mla_interface.py` | Python API |
| `setup.py` | Build configuration, NVCC flags, architecture targets |

## Build Flags

- `FLASH_MLA_DISABLE_SM100=1` — skip SM100 compilation (required for NVCC < 12.9)
- `FLASH_MLA_DISABLE_SM120=1` — skip SM120 compilation
- `FLASH_MLA_DISABLE_SM90=1` — skip SM90 compilation
- `FLASH_MLA_HOST_HAS_SM120` — enables `FLASH_MLA_SM120_MODE` in SM90 kernel config
