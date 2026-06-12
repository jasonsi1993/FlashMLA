import time
import dataclasses
from typing import Tuple, List, Dict, Optional
import copy

import rich.console
import rich.table

import torch
import kernelkit as kk

import flash_mla

import lib
from lib import TestParam
from lib import RawTestParamForDecode as RawTestParam
import ref

"""
Generate testcase for unit test
"""

def gen_testcase() -> List[RawTestParam]:
    correctness_cases = []
    corner_cases = []
    for d_qk in [576, 512]:
        for have_extra_k in ([False, True] if d_qk == 512 else [False]):
            for have_extra_topk_len in ([False, True] if have_extra_k else [False]):
                for have_topk_len in ([False, True] if d_qk == 512 else [False]):
                    for h_q in [64, 128]:
                        cur_correctness_cases = [
                            RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                                        have_topk_length=have_topk_len,
                                        enable_attn_sink=True,
                                        extra_s_k=extra_s_k,
                                        extra_topk=extra_topk,
                                        block_size=block_size,
                                        extra_block_size=extra_block_size,
                                        have_extra_topk_length=have_extra_topk_len,
                                        d_qk=d_qk,
                                        check_correctness=True,
                                        num_runs=0)
                            for (s_k, topk, block_size) in [
                                (512, 64, 2),
                                (512, 64, 64),
                                (512, 64, 69),
                                (1024, 576, 2),
                                (1024, 576, 61),
                                (2046, 2048, 2),
                                (2046, 2048, 64),
                                (2046, 2048, 576)
                            ]
                            for (extra_s_k, extra_topk, extra_block_size) in ([
                                (512, 64, 2),
                                (512, 64, 64),
                                (512, 64, 69),
                                (1024, 576, 2),
                                (1024, 576, 61),
                                (2046, 2048, 2),
                                (2046, 2048, 64),
                                (2046, 2048, 576)
                            ] if have_extra_k else [(None, None, None)])
                            for b in [4, 74, 321]
                            for s_q in [1, 3]
                            for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])
                        ]
                        correctness_cases.extend(cur_correctness_cases)

                        cur_corner_cases = [
                            RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                                        is_all_indices_invalid=is_all_indices_invalid,
                                        have_zero_seqlen_k=have_zero_seqlen_k,
                                        have_topk_length=have_topk_len,
                                        enable_attn_sink=enable_attn_sink,
                                        extra_s_k=extra_s_k,
                                        extra_topk=extra_topk,
                                        block_size=block_size,
                                        extra_block_size=extra_block_size,
                                        have_extra_topk_length=have_extra_topk_len,
                                        d_qk=d_qk,
                                        check_correctness=True,
                                        num_runs=0,
                            )
                            for (s_k, topk, block_size) in [
                                (512, 64, 61),
                                (650, 576, 53),
                            ]
                            for (extra_s_k, extra_topk, extra_block_size) in ([
                                (512, 64, 61),
                                (650, 576, 53),
                            ] if have_extra_k else [(None, None, None)])
                            for b in [4, 74, 321]
                            for s_q in [3]
                            for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])
                            for is_all_indices_invalid in [True, False]
                            for have_zero_seqlen_k in [True, False]
                            for enable_attn_sink in [True, False]
                            if (is_all_indices_invalid or have_zero_seqlen_k or enable_attn_sink)
                        ]
                        corner_cases.extend(cur_corner_cases)

    base_and_bszs = [
        # V3.2
        (RawTestParam(0, 128, 2, 1, 32768, True, topk=2048, d_qk=576), [2, 64, 74, 128]),
        # MODEL1 CONFIG1
        (RawTestParam(0, 64, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=512, block_size=256, extra_block_size=64), [2, 64, 74, 128, 74*2, 256]),
        # MODEL1 CONFIG2
        (RawTestParam(0, 128, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=64), [2, 64, 74, 128, 74*2, 256]),
        # MODEL1 CONFIG3
        (RawTestParam(0, 64, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=2, have_extra_topk_length=True), [2, 64, 74, 128, 74*2, 256]),
        # MODEL1 CONFIG4
        (RawTestParam(0, 128, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=2, have_extra_topk_length=True), [2, 64, 74, 128, 74*2, 256]),
    ]
    performance_cases = [
        # Production cases
        dataclasses.replace(base, b=b)
        for base, bszs in base_and_bszs
        for b in bszs
    ] + [
        # Peak perf cases
        RawTestParam(74*2, h_q, 2, 1, 32768, True, topk=16384, d_qk=d_qk)
        for h_q in [64, 128]
        for d_qk in [512, 576]
    ]

    return correctness_cases + corner_cases + performance_cases


@dataclasses.dataclass
class Result:
    is_correct: bool
    compute_memory_ratio: float
    time_usage_per_us: float
    splitkv_time_usage_us: float
    combine_time_usage_us: float
    achieved_tflops: float
    achieved_gBps: float

_counter = kk.Counter()

@torch.inference_mode()
def test_flash_mla(p: TestParam) -> Result:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()
    assert p.decode

    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = lib.generate_testcase_for_decode(p)

    tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
    def run_decode():
        return lib.run_flash_mla_decode(p, t, tile_scheduler_metadata, None)
    
    # We first run the kernel once to generate output data for the correctness test
    # We must do this first, otherwise when allocating tensors for storing answers,
    # it may re-use memory that contains the correct answer, leading to false positives
    if p.check_correctness:
        torch.cuda.synchronize()
        out_ans, lse_ans = run_decode()
        torch.cuda.synchronize()
        # torch.set_printoptions(profile='full')
        # print(tile_scheduler_metadata.tile_scheduler_metadata[:, :7])
    
    # We run the performance test before generating the answer for the correctness test to avoid interference
    performance_result = Result(True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if p.num_runs == 0:
        performance_result = Result(True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        try:
            result = kk.bench_kineto(run_decode, p.num_runs)

            splitkv_kernel_name = "flash_fwd_splitkv_mla_fp8_sparse_kernel"
            combine_kernel_name = "flash_fwd_mla_combine_kernel"

            kernel_time_usages_us: Dict[str, Optional[float]] = {}
            def pick_kernel_time_usage(kernel_name: str):
                t = [kernel_name in s for s in result.get_kernel_names()]
                if any(t):
                    assert sum(t) == 1
                    kernel_time_usages_us[kernel_name] = result.get_kernel_time(kernel_name) * 1e6
                else:
                    kernel_time_usages_us[kernel_name] = None
            pick_kernel_time_usage(splitkv_kernel_name)
            pick_kernel_time_usage(combine_kernel_name)

            def have_kernel(name: str):
                return kernel_time_usages_us[name] is not None

            if kk.is_using_profiling_tools() or result.is_using_nsys:
                e2e_time_usage_us = 1e6
            else:
                assert have_kernel(splitkv_kernel_name)
                if have_kernel(combine_kernel_name):
                    e2e_time_usage_us = result.get_e2e_time(splitkv_kernel_name, combine_kernel_name) * 1e6
                else:
                    e2e_time_usage_us = kernel_time_usages_us[splitkv_kernel_name]
            assert e2e_time_usage_us is not None

            flops_and_mem_vol = lib.count_flop_and_mem_vol_for_decode(p, t)
            e2e_time_usage_s = e2e_time_usage_us / 1e6
            theoritical_compute_memory_ratio = flops_and_mem_vol.flop / flops_and_mem_vol.mem_vol
            achieved_tflops = flops_and_mem_vol.flop / e2e_time_usage_s / 1e12
            achieved_gBps = flops_and_mem_vol.mem_vol / e2e_time_usage_s / 1e9
            def print_kernel_time_usage(name: str, short_name: str):
                if kernel_time_usages_us[name] is not None:
                    print(f'{short_name} time: {kernel_time_usages_us[name]:.1f} us')
            print(f'Compute/Memory: {theoritical_compute_memory_ratio:.2f}')
            print(f'Time (per): {e2e_time_usage_us:.1f} us')
            print_kernel_time_usage(splitkv_kernel_name, "Splitkv")
            print_kernel_time_usage(combine_kernel_name, "Combine")
            print(f'TFlops: {achieved_tflops:.1f}')
            print(f'GB/s: {achieved_gBps:.0f}')
            performance_result = Result(True, theoritical_compute_memory_ratio, e2e_time_usage_us, kernel_time_usages_us[splitkv_kernel_name] or 0.0, kernel_time_usages_us[combine_kernel_name] or 0.0, achieved_tflops, achieved_gBps)
        except Exception:
            performance_result = Result(True, 0.0, 1e6, 0.0, 0.0, 0.0, 0.0)
    
    is_correct = True
    if p.check_correctness:
        torch.cuda.synchronize()
        with torch.profiler.record_function("reference_flash_mla"):
            out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)

        is_out_correct = kk.check_is_allclose("out", out_ans, out_ref, abs_tol=1e-3, rel_tol=2.01/128, cos_diff_tol=5e-6)
        is_lse_correct = kk.check_is_allclose("lse", lse_ans, lse_ref, abs_tol=1e-6, rel_tol=8.01/65536)
        is_correct &= is_out_correct and is_lse_correct

    performance_result.is_correct = is_correct
    return performance_result


def _run_test_on_gpu(gpu_id, test_indices, all_testcases):
    """Run a subset of tests on a specific GPU."""
    import os, io, sys
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import torch
    torch.cuda.empty_cache()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.set_float32_matmul_precision('high')

    results = []
    for idx in test_indices:
        tc = all_testcases[idx]
        try:
            old_stdout = sys.stdout; sys.stdout = io.StringIO()
            r = test_flash_mla(tc)
            sys.stdout = old_stdout
            results.append((idx, r.is_correct, None))
        except Exception as e:
            sys.stdout = old_stdout
            results.append((idx, False, f'{type(e).__name__}: {str(e)[:60]}'))
    return gpu_id, results

def main():
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(32)

    raw_testcases = gen_testcase()
    testcases = [t.to_test_param() for t in raw_testcases]

    num_gpus = min(8, torch.cuda.device_count())
    print(f"{kk.colors['CYAN_BG']}{len(testcases)} testcases to run on {num_gpus} GPUs{kk.colors['CLEAR']}")

    # Distribute tests across GPUs (striped for load balance)
    gpu_assignments = [[] for _ in range(num_gpus)]
    for i in range(len(testcases)):
        gpu_assignments[i % num_gpus].append(i)

    # Run in parallel
    with mp.Pool(num_gpus) as pool:
        all_results = pool.starmap(_run_test_on_gpu,
            [(gpu, indices, testcases) for gpu, indices in enumerate(gpu_assignments)])

    # Merge results
    merged = {}
    for gpu_id, gpu_results in all_results:
        for idx, is_correct, error in gpu_results:
            merged[idx] = (is_correct, error)

    failed_cases = []
    for i in range(len(testcases)):
        is_correct, error = merged[i]
        if not is_correct:
            failed_cases.append((testcases[i], error))
            print(f"[{i+1}] FAIL: {error}" if error else f"[{i+1}] FAIL")

    if failed_cases:
        print(f"\n{len(failed_cases)}/{len(testcases)} FAILED:")
        for tc, err in failed_cases[:20]:
            print(f"  b={tc.decode.b} h={tc.h_q} topk={tc.topk} skv={tc.s_kv}: {err}")
        import sys; sys.exit(1)
    else:
        print(f"\nAll {len(testcases)} tests PASSED")
    

if __name__ == "__main__":
    main()
