#include "cuda_utils.cuh"
#include <torch/torch.h>

template <typename scalar_t>
__global__ void
convlogic_forward_kernel(const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
                         const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,
                         const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> selection,
                         torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> y, int stride_h,
                         int stride_w, int pad_h, int pad_w)
{
    const auto n = blockIdx.x;
    const auto c = blockIdx.y;
    const auto p = threadIdx.x;
    const auto tid_z = threadIdx.y;

    const auto H_in = x.size(2);
    const auto W_in = x.size(3);
    const auto H_out = y.size(2);
    const auto W_out = y.size(3);

    auto s1 = unpack_selection(selection[c][2 * p]);
    auto s2 = unpack_selection(selection[c][2 * p + 1]);

    s1.h -= pad_h;
    s1.w -= pad_w;
    s2.h -= pad_h;
    s2.w -= pad_w;

    scalar_t prob[15];
    for (auto f = 0; f < 15; ++f)
    {
        prob[f] = weights[c][p][f + 1];
    }

    for (auto out_y = tid_z; out_y < H_out; out_y += blockDim.y)
    {
        const int in_y1 = out_y * stride_h + s1.h;
        const int in_y2 = out_y * stride_h + s2.h;

        for (auto out_x = 0; out_x < W_out; ++out_x)
        {
            const int in_x1 = out_x * stride_w + s1.w;
            const int in_x2 = out_x * stride_w + s2.w;

            scalar_t x1 = 0, x2 = 0;

            if (in_y1 >= 0 && in_y1 < H_in && in_x1 >= 0 && in_x1 < W_in)
                x1 = x[n][s1.ch][in_y1][in_x1];

            if (in_y2 >= 0 && in_y2 < H_in && in_x2 >= 0 && in_x2 < W_in)
                x2 = x[n][s2.ch][in_y2][in_x2];

            scalar_t y_val = prob[0] * (x1 * x2) + prob[1] * (x1 - x1 * x2) + prob[2] * x1 + prob[3] * (x2 - x1 * x2) +
                             prob[4] * x2 + prob[5] * (x1 + x2 - scalar_t(2) * x1 * x2) +
                             prob[6] * (x1 + x2 - x1 * x2) + prob[7] * (scalar_t(1) - (x1 + x2 - x1 * x2)) +
                             prob[8] * (scalar_t(1) - (x1 + x2 - scalar_t(2) * x1 * x2)) +
                             prob[9] * (scalar_t(1) - x2) + prob[10] * (scalar_t(1) - x2 + x1 * x2) +
                             prob[11] * (scalar_t(1) - x1) + prob[12] * (scalar_t(1) - x1 + x1 * x2) +
                             prob[13] * (scalar_t(1) - x1 * x2) + prob[14];

            y[n][c][out_y][out_x][p] = y_val;
        }
    }
}

torch::Tensor convlogic_cuda_forward(torch::Tensor x, torch::Tensor weights, torch::Tensor selection, int stride_h,
                                     int stride_w, int pad_h, int pad_w, int kernel_h, int kernel_w)
{
    CHECK_INPUT(x);
    CHECK_INPUT(weights);
    CHECK_INPUT(selection);

    auto N = x.size(0);
    auto H_in = x.size(2);
    auto W_in = x.size(3);
    auto C_out = weights.size(0);
    auto H_out = (H_in + 2 * pad_h - kernel_h) / stride_h + 1;
    auto W_out = (W_in + 2 * pad_w - kernel_w) / stride_w + 1;

    constexpr auto p_total = 4;
    constexpr auto threads_z = 8;

    auto y = torch::empty({N, C_out, H_out, W_out, p_total}, x.options());

    CHECK_CUDA_BLOCK_LIMIT(p_total * threads_z, "convlogic_forward_kernel");
    dim3 grid(N, C_out);
    dim3 block(p_total, threads_z);

    AT_DISPATCH_FLOATING_TYPES(
        x.scalar_type(), "convlogic_cuda_forward",
        (
            [&]
            {
                convlogic_forward_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                    weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    selection.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                    y.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(), stride_h, stride_w, pad_h, pad_w);
            }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return y;
}

template <typename scalar_t>
__global__ void
convlogic_backward_weight_kernel(const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_y,
                                 const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
                                 const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> selection,
                                 torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_weight,
                                 int stride_h, int stride_w, int pad_h, int pad_w)
{
    const auto n = blockIdx.x;
    const auto c_out = blockIdx.y;
    const auto p = threadIdx.x;
    const auto tid_y = threadIdx.y;

    constexpr auto num_gates = 15;

    // Use char aliasing trick to avoid extern templated type problem
    extern __shared__ char smem_raw[];
    scalar_t *smem = reinterpret_cast<scalar_t *>(smem_raw);

    scalar_t *shared_ptr = smem + (p * blockDim.y + tid_y) * num_gates;

    // Init local accumulators
    scalar_t gate_vals[num_gates] = {0};

    // Selection info
    auto s1 = unpack_selection(selection[c_out][2 * p]);
    auto s2 = unpack_selection(selection[c_out][2 * p + 1]);

    auto H_in = x.size(2);
    auto W_in = x.size(3);
    auto H_out = grad_y.size(2);
    auto W_out = grad_y.size(3);

    for (auto out_y = tid_y; out_y < H_out; out_y += blockDim.y)
    {
        int in_y1 = out_y * stride_h + s1.h;
        int in_y2 = out_y * stride_h + s2.h;

        for (auto out_x = 0; out_x < W_out; ++out_x)
        {
            int in_x1 = out_x * stride_w + s1.w;
            int in_x2 = out_x * stride_w + s2.w;

            scalar_t gy = grad_y[n][c_out][out_y][out_x][p];

            scalar_t x1 = x[n][s1.ch][in_y1][in_x1];
            scalar_t x2 = x[n][s2.ch][in_y2][in_x2];

            gate_vals[0] += gy * (x1 * x2);
            gate_vals[1] += gy * (x1 - x1 * x2);
            gate_vals[2] += gy * x1;
            gate_vals[3] += gy * (x2 - x1 * x2);
            gate_vals[4] += gy * x2;
            gate_vals[5] += gy * (x1 + x2 - scalar_t(2) * x1 * x2);
            gate_vals[6] += gy * (x1 + x2 - x1 * x2);
            gate_vals[7] += gy * (scalar_t(1) - (x1 + x2 - x1 * x2));
            gate_vals[8] += gy * (scalar_t(1) - (x1 + x2 - scalar_t(2) * x1 * x2));
            gate_vals[9] += gy * (scalar_t(1) - x2);
            gate_vals[10] += gy * (scalar_t(1) - x2 + x1 * x2);
            gate_vals[11] += gy * (scalar_t(1) - x1);
            gate_vals[12] += gy * (scalar_t(1) - x1 + x1 * x2);
            gate_vals[13] += gy * (scalar_t(1) - x1 * x2);
            gate_vals[14] += gy;
        }
    }

    // Write partials to shared memory
    for (auto g = 0; g < num_gates; ++g)
    {
        shared_ptr[g] = gate_vals[g];
    }

    __syncthreads();

    // First thread per (p, gate) reduces across z
    if (tid_y == 0)
    {
        for (auto g = 0; g < num_gates; ++g)
        {
            scalar_t acc = 0;
            for (auto y = 0; y < blockDim.y; ++y)
            {
                acc += smem[(p * blockDim.y + y) * num_gates + g];
            }
            grad_weight[n][c_out][p][g] = acc;
        }
    }
}

torch::Tensor convlogic_cuda_backward_weight(torch::Tensor grad_y, torch::Tensor x, torch::Tensor selection,
                                             int stride_h, int stride_w, int pad_h, int pad_w)
{
    CHECK_INPUT(grad_y);
    CHECK_INPUT(x);
    CHECK_INPUT(selection);

    const auto N = x.size(0);
    const auto H_in = x.size(2);
    const auto W_in = x.size(3);
    const auto C_out = grad_y.size(1);

    constexpr auto p_total = 4;
    constexpr auto gates_total = 16;
    constexpr auto threads_y = 8;

    auto grad_weight_partial =
        torch::empty({N, C_out, p_total, gates_total - 1}, torch::dtype(x.dtype()).device(x.device()));

    auto x_padded = torch::constant_pad_nd(x, {pad_w, pad_w, pad_h, pad_h}, 0);

    CHECK_CUDA_BLOCK_LIMIT(p_total * threads_y, "convlogic_cuda_backward_weight");
    dim3 grid(N, C_out);
    dim3 block(p_total, threads_y);

    AT_DISPATCH_FLOATING_TYPES(
        x.scalar_type(), "convlogic_cuda_backward_weight",
        (
            [&]
            {
                size_t shared_mem = p_total * (gates_total - 1) * threads_y * sizeof(scalar_t);
                CHECK_SHARED_MEMORY(shared_mem);

                convlogic_backward_weight_kernel<scalar_t>
                    <<<grid, block, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
                        grad_y.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                        x_padded.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                        selection.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                        grad_weight_partial.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), stride_h,
                        stride_w, pad_h, pad_w);
            }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto grad_weight_summed = grad_weight_partial.sum(0);
    auto grad_weight = torch::empty({C_out, 4, 16}, grad_weight_summed.options());
    grad_weight.slice(2, 0, 1).zero_();
    grad_weight.slice(2, 1, 16).copy_(grad_weight_summed);

    return grad_weight;
}

template <typename scalar_t>
__global__ void
convlogic_backward_x_kernel(const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_y,
                            const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
                            const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,
                            const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> selection,
                            torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_x, int stride_h,
                            int stride_w, int pad_h, int pad_w)
{
    const auto n = blockIdx.x;
    const auto c_out = blockIdx.y;
    const auto p = threadIdx.x;
    const auto tid_y = threadIdx.y;

    auto s1 = unpack_selection(selection[c_out][2 * p]);
    auto s2 = unpack_selection(selection[c_out][2 * p + 1]);

    s1.h -= pad_h;
    s1.w -= pad_w;
    s2.h -= pad_h;
    s2.w -= pad_w;

    scalar_t prob[15];
    for (auto f = 0; f < 15; ++f)
    {
        prob[f] = weights[c_out][p][f + 1];
    }

    const auto H_in = x.size(2);
    const auto W_in = x.size(3);
    const auto H_out = grad_y.size(2);
    const auto W_out = grad_y.size(3);

    for (auto out_y = tid_y; out_y < H_out; out_y += blockDim.y)
    {
        const int in_y1 = out_y * stride_h + s1.h;
        const int in_y2 = out_y * stride_h + s2.h;

        for (auto out_x = 0; out_x < W_out; ++out_x)
        {
            const int in_x1 = out_x * stride_w + s1.w;
            const int in_x2 = out_x * stride_w + s2.w;

            scalar_t grad_y_val = grad_y[n][c_out][out_y][out_x][p];
            bool valid1 = in_y1 >= 0 && in_y1 < H_in && in_x1 >= 0 && in_x1 < W_in;
            bool valid2 = in_y2 >= 0 && in_y2 < H_in && in_x2 >= 0 && in_x2 < W_in;

            scalar_t x1 = (valid1) ? x[n][s1.ch][in_y1][in_x1] : 0;
            scalar_t x2 = (valid2) ? x[n][s2.ch][in_y2][in_x2] : 0;

            if (valid1)
            {
                const auto dy_dx1 = prob[0] * x2 + prob[1] * (scalar_t(1) - x2) + prob[2] + prob[3] * (-x2) +
                                    prob[5] * (scalar_t(1) - scalar_t(2) * x2) + prob[6] * (scalar_t(1) - x2) +
                                    prob[7] * (x2 - scalar_t(1)) + prob[8] * (scalar_t(2) * x2 - scalar_t(1)) +
                                    prob[10] * x2 - prob[11] + prob[12] * (x2 - scalar_t(1)) + prob[13] * (-x2);
                atomicAdd(&grad_x[n][s1.ch][in_y1][in_x1], dy_dx1 * grad_y_val);
            }

            if (valid2)
            {
                const auto dy_dx2 = prob[0] * x1 + prob[1] * (-x1) + prob[3] * (scalar_t(1) - x1) + prob[4] +
                                    prob[5] * (scalar_t(1) - scalar_t(2) * x1) + prob[6] * (scalar_t(1) - x1) +
                                    prob[7] * (x1 - scalar_t(1)) + prob[8] * (scalar_t(2) * x1 - scalar_t(1)) -
                                    prob[9] + prob[10] * (x1 - scalar_t(1)) + prob[12] * x1 + prob[13] * (-x1);
                atomicAdd(&grad_x[n][s2.ch][in_y2][in_x2], dy_dx2 * grad_y_val);
            }
        }
    }
}

torch::Tensor convlogic_cuda_backward_x(torch::Tensor grad_y, torch::Tensor x, torch::Tensor weights,
                                        torch::Tensor selection, int stride_h, int stride_w, int pad_h, int pad_w)
{
    CHECK_INPUT(grad_y);
    CHECK_INPUT(x);
    CHECK_INPUT(weights);
    CHECK_INPUT(selection);

    const auto N = x.size(0);
    const auto C_out = grad_y.size(1);

    constexpr const auto p_total = 4;
    constexpr const auto threads_y = 8;

    auto grad_x = torch::zeros_like(x);

    CHECK_CUDA_BLOCK_LIMIT(p_total * threads_y, "convlogic_backward_x_kernel");
    dim3 grid(N, C_out);
    dim3 block(p_total, threads_y);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "convlogic_backward_x_kernel",
                               (
                                   [&]
                                   {
                                       convlogic_backward_x_kernel<scalar_t>
                                           <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                               grad_y.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                                               x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               selection.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                               grad_x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               stride_h, stride_w, pad_h, pad_w);
                                   }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return grad_x;
}
