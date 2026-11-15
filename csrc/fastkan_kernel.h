// =========================================================================
// Объявления CUDA-ядер FastKAN (fused RBF basis + SiLU).
// Определения — в fastkan_kernel.cu, C++ обёртки — в bindings.cpp.
// =========================================================================
#pragma once

#include <torch/extension.h>

// Forward: fused RBF basis expansion + SiLU.
// x: [B, C, T] (логически [B, C, 1, T]), grid: [G]
// rbf_out: [B, C * G, T], silu_out: [B, C, T]
template <typename scalar_t>
__global__ void fastkan_rbf_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ grid,
    scalar_t* __restrict__ rbf_out,
    scalar_t* __restrict__ silu_out,
    const int B,
    const int C,
    const int T,
    const int G,
    const float inv_denom
);

// Backward: градиент по входу x от обеих веток (grad_rbf и grad_silu).
template <typename scalar_t>
__global__ void fastkan_rbf_backward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ grid,
    const scalar_t* __restrict__ rbf_cache,
    const scalar_t* __restrict__ grad_rbf,
    const scalar_t* __restrict__ grad_silu,
    scalar_t* __restrict__ grad_x,
    const int B,
    const int C,
    const int T,
    const int G,
    const float inv_denom
);
