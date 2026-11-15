// =========================================================================
// C++ обёртки и pybind11-биндинги для FastKAN CUDA-ядер.
//
// Перенесено из FastKAN_CUDA_Kernel/fastkan_kernel.cu (легаси-каталог).
// Ядра объявлены в fastkan_kernel.h, определены в fastkan_kernel.cu.
// =========================================================================
#include <torch/extension.h>

#include "fastkan_kernel.h"

// =========================================================================
// C++ Обертки для PyTorch Autograd
// =========================================================================
std::vector<torch::Tensor> fastkan_basis_forward_cuda(
    torch::Tensor x,
    torch::Tensor grid,
    float inv_denom
) {
    // x: [B, C, 1, T] -> работаем как с [B, C, T]
    const int B = x.size(0);
    const int C = x.size(1);
    const int T = x.size(3);
    const int G = grid.size(0);

    auto rbf_out = torch::empty({B, C * G, 1, T}, x.options());
    auto silu_out = torch::empty_like(x);

    const int total = B * C * T;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fastkan_rbf_forward_cuda", ([&] {
            fastkan_rbf_forward_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                grid.data_ptr<scalar_t>(),
                rbf_out.data_ptr<scalar_t>(),
                silu_out.data_ptr<scalar_t>(),
                B, C, T, G, inv_denom
            );
        })
    );

    return {rbf_out, silu_out};
}

torch::Tensor fastkan_basis_backward_cuda(
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor rbf_cache,
    torch::Tensor grad_rbf,
    torch::Tensor grad_silu,
    float inv_denom
) {
    const int B = x.size(0);
    const int C = x.size(1);
    const int T = x.size(3);
    const int G = grid.size(0);

    auto grad_x = torch::empty_like(x);

    const int total = B * C * T;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fastkan_rbf_backward_cuda", ([&] {
            fastkan_rbf_backward_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                grid.data_ptr<scalar_t>(),
                rbf_cache.data_ptr<scalar_t>(),
                grad_rbf.data_ptr<scalar_t>(),
                grad_silu.data_ptr<scalar_t>(),
                grad_x.data_ptr<scalar_t>(),
                B, C, T, G, inv_denom
            );
        })
    );

    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fastkan_basis_forward_cuda, "FastKAN Fused Basis Forward (CUDA)");
    m.def("backward", &fastkan_basis_backward_cuda, "FastKAN Fused Basis Backward (CUDA)");
}
