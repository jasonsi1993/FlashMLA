#include "../kernel.cuh"

namespace sm120::decode::sparse_fp8 {

template void run_sm120_sparse_decode_kernel<ModelType::V32, 128>(const SparseAttnDecodeParams &params);

}
