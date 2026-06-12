#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace sm120::decode::sparse_fp8 {

struct DebugProbes {
    static constexpr int IDX_OFFSET = 0;
    static constexpr int K_OFFSET = 64;
    static constexpr int QK_OFFSET = K_OFFSET + 64*576;
    static constexpr int S_OFFSET = QK_OFFSET + 64*64;
    static constexpr int V_OFFSET = S_OFFSET + 64*512;
    static constexpr int O_OFFSET = V_OFFSET + 64*512;
    static constexpr int TOTAL = O_OFFSET + 64*512;
};

// Host-side management
void debug_alloc();                     // allocate GPU buffer
void debug_free();                      // free GPU buffer
float* debug_get_ptr();                 // get GPU buffer pointer
int* debug_get_indices_ptr();           // get GPU indices buffer pointer
bool debug_is_enabled();                // check if debug is enabled
void debug_copy_to_host(float* host, size_t count);  // copy GPU→CPU
void debug_copy_indices_to_host(int* host, size_t count);

}  // namespace sm120::decode::sparse_fp8
