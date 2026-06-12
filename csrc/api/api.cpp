#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"
#include "sm120/decode/sparse_fp8/debug.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);

    // Debug probes
    m.def("debug_alloc", []() {
        sm120::decode::sparse_fp8::debug_alloc();
    });
    m.def("debug_free", []() {
        sm120::decode::sparse_fp8::debug_free();
    });
    m.def("debug_get_data", []() {
        size_t size = sm120::decode::sparse_fp8::DebugProbes::TOTAL;
        auto tensor = torch::empty({static_cast<int64_t>(size)}, torch::TensorOptions().dtype(torch::kFloat32));
        sm120::decode::sparse_fp8::debug_copy_to_host(tensor.data_ptr<float>(), size);
        return tensor;
    });
    m.def("debug_get_indices", []() {
        auto tensor = torch::empty({64}, torch::TensorOptions().dtype(torch::kInt32));
        sm120::decode::sparse_fp8::debug_copy_indices_to_host(tensor.data_ptr<int>(), 64);
        return tensor;
    });
}
