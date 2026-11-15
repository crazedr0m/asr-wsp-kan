// =========================================================================
// FastKAN fused CUDA kernels: RBF basis expansion + SiLU (forward/backward).
//
// Перенесено из FastKAN_CUDA_Kernel/fastkan_kernel.cu (легаси-каталог).
// Здесь только CUDA-ядра; C++ обёртки и pybind11-биндинги — в bindings.cpp.
// =========================================================================
#include <cuda.h>
#include <cuda_runtime.h>

#include "fastkan_kernel.h"

// =========================================================================
// 1. FORWARD KERNEL: Fused RBF Basis Expansion + SiLU
// Избегаем создания промежуточного тензора [B, C, Grid, T] в PyTorch
// и считаем все num_grids значений прямо в регистрах потока GPU.
// =========================================================================
template <typename scalar_t>
__global__ void fastkan_rbf_forward_kernel(
    const scalar_t* __restrict__ x,          // [B, C, T]
    const scalar_t* __restrict__ grid,       // [G]
    scalar_t* __restrict__ rbf_out,          // [B, C * G, T]
    scalar_t* __restrict__ silu_out,         // [B, C, T]
    const int B,
    const int C,
    const int T,
    const int G,
    const float inv_denom
) {
    // Каждый поток обрабатывает один элемент (b, c, t)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * T;
    if (idx >= total) return;

    int t = idx % T;
    int tmp = idx / T;
    int c = tmp % C;
    int b = tmp / C;

    float val = static_cast<float>(x[idx]);

    // Считаем SiLU(x) = x / (1 + exp(-x)) за один проход
    float sig = 1.0f / (1.0f + expf(-val));
    silu_out[idx] = static_cast<scalar_t>(val * sig);

    // Считаем G гауссиан и записываем в [b, c * G + g, t]
    int base_out_offset = (b * C * G + c * G) * T + t;
    for (int g = 0; g < G; ++g) {
        float g_val = static_cast<float>(grid[g]);
        float diff = (val - g_val) * inv_denom;
        float rbf = expf(-diff * diff);
        rbf_out[base_out_offset + g * T] = static_cast<scalar_t>(rbf);
    }
}

// =========================================================================
// 2. BACKWARD KERNEL: Градиент по входу x от обеих веток (grad_rbf и grad_silu)
// d(exp(-diff^2))/dx = -2 * diff * inv_denom * exp(-diff^2)
// d(x * sig(x))/dx = sig(x) * (1 + x * (1 - sig(x)))
// =========================================================================
template <typename scalar_t>
__global__ void fastkan_rbf_backward_kernel(
    const scalar_t* __restrict__ x,          // [B, C, T]
    const scalar_t* __restrict__ grid,       // [G]
    const scalar_t* __restrict__ rbf_cache,  // [B, C * G, T] (переиспользуем посчитанный RBF!)
    const scalar_t* __restrict__ grad_rbf,   // [B, C * G, T]
    const scalar_t* __restrict__ grad_silu,  // [B, C, T]
    scalar_t* __restrict__ grad_x,           // [B, C, T]
    const int B,
    const int C,
    const int T,
    const int G,
    const float inv_denom
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * T;
    if (idx >= total) return;

    int t = idx % T;
    int tmp = idx / T;
    int c = tmp % C;
    int b = tmp / C;

    float val = static_cast<float>(x[idx]);

    // 1. Градиент от SiLU ветки
    float sig = 1.0f / (1.0f + expf(-val));
    float d_silu = sig * (1.0f + val * (1.0f - sig));
    float acc_grad = static_cast<float>(grad_silu[idx]) * d_silu;

    // 2. Суммируем градиенты от всех G точек сетки RBF
    int base_offset = (b * C * G + c * G) * T + t;
    float factor = -2.0f * inv_denom * inv_denom;

    for (int g = 0; g < G; ++g) {
        int offset = base_offset + g * T;
        float g_val = static_cast<float>(grid[g]);
        float rbf_val = static_cast<float>(rbf_cache[offset]);
        float gr = static_cast<float>(grad_rbf[offset]);

        // d(rbf)/dx = -2 * ((val - g_val) * inv_denom) * inv_denom * rbf_val
        float d_rbf = factor * (val - g_val) * rbf_val;
        acc_grad += gr * d_rbf;
    }

    grad_x[idx] = static_cast<scalar_t>(acc_grad);
}

// =========================================================================
// Явные инстанцирования шаблонов (используются из bindings.cpp)
// =========================================================================
#define INSTANTIATE_FORWARD(TYPE)                                              \
    template __global__ void fastkan_rbf_forward_kernel<TYPE>(                 \
        const TYPE*, const TYPE*, TYPE*, TYPE*, int, int, int, int, float);

#define INSTANTIATE_BACKWARD(TYPE)                                             \
    template __global__ void fastkan_rbf_backward_kernel<TYPE>(                \
        const TYPE*, const TYPE*, const TYPE*, const TYPE*, const TYPE*,       \
        TYPE*, int, int, int, int, float);

INSTANTIATE_FORWARD(float)
INSTANTIATE_FORWARD(at::Half)
INSTANTIATE_FORWARD(at::BFloat16)

INSTANTIATE_BACKWARD(float)
INSTANTIATE_BACKWARD(at::Half)
INSTANTIATE_BACKWARD(at::BFloat16)

#undef INSTANTIATE_FORWARD
#undef INSTANTIATE_BACKWARD
