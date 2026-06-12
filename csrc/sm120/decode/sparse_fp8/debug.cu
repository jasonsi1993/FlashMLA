#include "debug.h"

namespace sm120::decode::sparse_fp8 {

static float* g_debug_buffer = nullptr;
static int* g_debug_indices = nullptr;
static bool g_debug_enabled = false;

void debug_alloc() {
    if (!g_debug_buffer) {
        cudaMalloc(&g_debug_buffer, DebugProbes::TOTAL * sizeof(float));
        cudaMalloc(&g_debug_indices, 64 * sizeof(int));
    }
    // Always zero buffers
    if (g_debug_buffer) {
        cudaMemset(g_debug_buffer, 0, DebugProbes::TOTAL * sizeof(float));
    }
    if (g_debug_indices) {
        cudaMemset(g_debug_indices, 0, 64 * sizeof(int));
    }
    g_debug_enabled = true;
}

void debug_free() {
    if (g_debug_buffer) {
        cudaFree(g_debug_buffer);
        g_debug_buffer = nullptr;
    }
    if (g_debug_indices) {
        cudaFree(g_debug_indices);
        g_debug_indices = nullptr;
    }
    g_debug_enabled = false;
}

float* debug_get_ptr() {
    return g_debug_buffer;
}

int* debug_get_indices_ptr() {
    return g_debug_indices;
}

bool debug_is_enabled() {
    return g_debug_enabled;
}

void debug_copy_to_host(float* host, size_t count) {
    if (g_debug_buffer && count > 0) {
        cudaMemcpy(host, g_debug_buffer,
                   std::min(count, (size_t)DebugProbes::TOTAL) * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
}

void debug_copy_indices_to_host(int* host, size_t count) {
    if (g_debug_indices && count > 0) {
        cudaMemcpy(host, g_debug_indices,
                   std::min(count, (size_t)64) * sizeof(int),
                   cudaMemcpyDeviceToHost);
    }
}

}  // namespace sm120::decode::sparse_fp8
