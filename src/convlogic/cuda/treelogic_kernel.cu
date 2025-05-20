#include "cuda_utils.cuh"
#include "cuda_vec.cuh"

template <typename scalar_t>
__global__ void
treelogic_forward_kernel(const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> x,
                         const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,
                         torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> y)
{
    const auto n = blockIdx.x;
    const auto c = blockIdx.y;
    const auto p = threadIdx.x;
    const auto tid_y = threadIdx.y;

    const auto H_out = y.size(2);
    const auto W_out = y.size(3);

    scalar_t prob[15];
    for (auto f = 0; f < 15; ++f)
    {
        prob[f] = weights[c][p][f + 1];
    }

    for (auto out_y = tid_y; out_y < H_out; out_y += blockDim.y)
    {

        for (auto out_x = 0; out_x < W_out; ++out_x)
        {
            scalar_t x1, x2;
            vec2_load(&x[n][c][out_y][out_x][2 * p], x1, x2);

            scalar_t y_val = prob[0] * (x1 * x2) + prob[1] * (x1 - x1 * x2) + prob[2] * x1 + prob[3] * (x2 - x1 * x2) +
                             prob[4] * x2 + prob[5] * (x1 + x2 - scalar_t(2) * x1 * x2) +
                             prob[6] * (x1 + x2 - x1 * x2) + prob[7] * (scalar_t(1) - (x1 + x2 - x1 * x2)) +
                             prob[8] * (scalar_t(1) - (x1 + x2 - scalar_t(2) * x1 * x2)) +
                             prob[9] * (scalar_t(1) - x2) + prob[10] * (scalar_t(1) - x2 + x1 * x2) +
                             prob[11] * (scalar_t(1) - x1) + prob[12] * (scalar_t(1) - x1 + x1 * x2) +
                             prob[13] * (scalar_t(1) - x1 * x2) + prob[14];
            // Note: prob[0] term (multiplied by 0) is omitted

            y[n][c][out_y][out_x][p] = y_val;
        }
    }
}

torch::Tensor treelogic_cuda_forward(torch::Tensor x, torch::Tensor weights)
{
    CHECK_INPUT(x);
    CHECK_INPUT(weights);

    TORCH_CHECK(x.dim() == 5, "Expected input tensor x to be 5D (N, C, H, W, 2*Xout), but got ", x.dim());

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    const auto Xin = x.size(4);
    const auto Xout = weights.size(1);

    TORCH_CHECK(Xin == 2 * Xout, "Expected input x.size(4) (Xin) to equal 2*Xout (", 2 * Xout, "), but got ", Xin);

    auto threads_y = 32 / Xout;

    auto y = torch::empty({N, C, H, W, Xout}, torch::dtype(x.dtype()).device(x.device()));

    CHECK_CUDA_BLOCK_LIMIT(Xout * threads_y, "treelogic_cuda_forward");
    dim3 grid(N, C);
    dim3 block(Xout, threads_y);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "treelogic_cuda_forward",
                               (
                                   [&]
                                   {
                                       treelogic_forward_kernel<scalar_t>
                                           <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                               x.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                                               weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               y.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>());
                                   }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return y;
}

template <typename scalar_t>
__global__ void
treelogic_backward_weight_kernel(const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_y,
                                 const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> x,
                                 torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_weight)
{
    const int n = blockIdx.x;
    const int c = blockIdx.y;
    const int p = threadIdx.x;
    const int tid_y = threadIdx.y;

    int H = x.size(2);
    int W = x.size(3);
    int Xout = grad_y.size(4);

    const int num_gates = 15;

    // TODO: Make shared memory PackedTensorAccessor64
    extern __shared__ char smem_raw[];
    scalar_t *smem = reinterpret_cast<scalar_t *>(smem_raw);

    scalar_t *shared_ptr = smem + (p * blockDim.y + tid_y) * num_gates;

    scalar_t gate_vals[num_gates] = {0};

    for (int out_y = tid_y; out_y < H; out_y += blockDim.y)
    {
        for (int out_x = 0; out_x < W; ++out_x)
        {
            scalar_t gy_val = grad_y[n][c][out_y][out_x][p];
            scalar_t x1, x2;
            vec2_load(&x[n][c][out_y][out_x][2 * p], x1, x2);

            gate_vals[0] += gy_val * (x1 * x2);
            gate_vals[1] += gy_val * (x1 - x1 * x2);
            gate_vals[2] += gy_val * x1;
            gate_vals[3] += gy_val * (x2 - x1 * x2);
            gate_vals[4] += gy_val * x2;
            gate_vals[5] += gy_val * (x1 + x2 - 2 * x1 * x2);
            gate_vals[6] += gy_val * (x1 + x2 - x1 * x2);
            gate_vals[7] += gy_val * (1 - (x1 + x2 - x1 * x2));
            gate_vals[8] += gy_val * (1 - (x1 + x2 - 2 * x1 * x2));
            gate_vals[9] += gy_val * (1 - x2);
            gate_vals[10] += gy_val * (1 - x2 + x1 * x2);
            gate_vals[11] += gy_val * (1 - x1);
            gate_vals[12] += gy_val * (1 - x1 + x1 * x2);
            gate_vals[13] += gy_val * (1 - x1 * x2);
            gate_vals[14] += gy_val;
        }
    }

    for (int g = 0; g < num_gates; ++g)
    {
        shared_ptr[g] = gate_vals[g];
    }

    __syncthreads();

    if (tid_y == 0)
    {
        for (int g = 0; g < num_gates; ++g)
        {
            scalar_t acc = 0;
            for (int y = 0; y < blockDim.y; ++y)
            {
                acc += smem[p * num_gates * blockDim.y + y * num_gates + g];
            }

            grad_weight[n][c][p][g] = acc;
        }
    }
}

torch::Tensor treelogic_cuda_backward_weight(torch::Tensor grad_y, torch::Tensor x)
{
    CHECK_INPUT(grad_y);
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 5, "Expected input tensor x to be 5D (N, C, H, W, 2*Xout), but got ", x.dim());

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto Xin = x.size(4);
    const auto Xout = Xin / 2;

    TORCH_CHECK(Xin == 2 * Xout, "Expected input x.size(4) (Xin) to equal 2*Xout (", 2 * Xout, "), but got ", Xin);
    TORCH_CHECK(Xout <= 4, "Xout > 4 is not supported");

    auto grad_weight_partial = torch::empty({N, C, Xout, 16 - 1}, torch::dtype(x.dtype()).device(x.device()));

    auto threads_y = 32 / Xout;

    CHECK_CUDA_BLOCK_LIMIT(Xout * threads_y, "treelogic_backward_weight_kernel");
    dim3 grid(N, C);
    dim3 block(Xout, threads_y);

    AT_DISPATCH_FLOATING_TYPES(
        x.scalar_type(), "treelogic_backward_weight_kernel",
        (
            [&]
            {
                size_t shared_mem = Xout * threads_y * 15 * sizeof(scalar_t);
                CHECK_SHARED_MEMORY(shared_mem);

                treelogic_backward_weight_kernel<scalar_t>
                    <<<grid, block, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
                        grad_y.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                        x.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                        grad_weight_partial.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
            }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto grad_weight_summed = grad_weight_partial.sum(0);
    auto grad_weight = torch::empty({C, Xout, 16}, grad_weight_summed.options());
    grad_weight.slice(2, 0, 1).zero_();
    grad_weight.slice(2, 1, 16).copy_(grad_weight_summed);

    return grad_weight;
}

template <typename scalar_t>
__global__ void
treelogic_backward_x_kernel(const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_y,
                            const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> x,
                            const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,
                            torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_x)
{
    const auto n = blockIdx.x;
    const auto c = blockIdx.y;
    const auto p = threadIdx.x;
    const auto tid_y = threadIdx.y;

    const auto H_out = grad_y.size(2);
    const auto W_out = grad_y.size(3);

    scalar_t prob[15];
    for (auto f = 0; f < 15; ++f)
    {
        prob[f] = weights[c][p][f + 1];
    }

    for (auto out_y = tid_y; out_y < H_out; out_y += blockDim.y)
    {
        for (auto out_x = 0; out_x < W_out; ++out_x)
        {
            scalar_t x1, x2;
            vec2_load(&x[n][c][out_y][out_x][2 * p], x1, x2);
            scalar_t grad_y_val = grad_y[n][c][out_y][out_x][p];

            const auto dy_dx1 = prob[0] * x2 + prob[1] * (scalar_t(1) - x2) + prob[2] + prob[3] * (-x2) +
                                prob[5] * (scalar_t(1) - scalar_t(2) * x2) + prob[6] * (scalar_t(1) - x2) +
                                prob[7] * (x2 - scalar_t(1)) + prob[8] * (scalar_t(2) * x2 - scalar_t(1)) +
                                prob[10] * x2 - prob[11] + prob[12] * (x2 - scalar_t(1)) + prob[13] * (-x2);

            const auto dy_dx2 = prob[0] * x1 + prob[1] * (-x1) + prob[3] * (scalar_t(1) - x1) + prob[4] +
                                prob[5] * (scalar_t(1) - scalar_t(2) * x1) + prob[6] * (scalar_t(1) - x1) +
                                prob[7] * (x1 - scalar_t(1)) + prob[8] * (scalar_t(2) * x1 - scalar_t(1)) - prob[9] +
                                prob[10] * (x1 - scalar_t(1)) + prob[12] * x1 + prob[13] * (-x1);

            vec2_store(&grad_x[n][c][out_y][out_x][2 * p], grad_y_val * dy_dx1, grad_y_val * dy_dx2);
        }
    }
}

torch::Tensor treelogic_cuda_backward_x(torch::Tensor grad_y, torch::Tensor x, torch::Tensor weights)
{
    CHECK_INPUT(grad_y);
    CHECK_INPUT(x);
    CHECK_INPUT(weights);

    const auto N = x.size(0);
    const auto C = x.size(1);
    const int Xout = weights.size(1);

    auto threads_z = 32 / Xout;

    auto grad_x = torch::empty_like(x);

    CHECK_CUDA_BLOCK_LIMIT(Xout * threads_z, "treelogic_backward_x_kernel");
    dim3 grid(N, C);
    dim3 block(Xout, threads_z);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "treelogic_backward_x_kernel",
                               (
                                   [&]
                                   {
                                       treelogic_backward_x_kernel<scalar_t>
                                           <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                               grad_y.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                                               x.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                                               weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               grad_x.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>());
                                   }));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return grad_x;
}
