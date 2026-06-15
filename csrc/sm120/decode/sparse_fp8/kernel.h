#pragma once

#include "params.h"

namespace sm120::decode::sparse_fp8 {

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_sm120_sparse_decode_kernel(const SparseAttnDecodeParams &params);

}
