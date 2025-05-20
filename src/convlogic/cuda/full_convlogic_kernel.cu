#include "cuda_config.cuh"
#include "cuda_utils.cuh"
#include "cuda_vec.cuh"
#include <torch/torch.h>

#define CONV_START 0
#define TREE1_START 4
#define TREE2_START 6

template <typename scalar_t>
__global__ void
full_convlogic_forward_kernel(const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
                              const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,
                              const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> selection,
                              torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> y, int stride_h,
                              int stride_w, int pad_h, int pad_w)
{
    const auto n = blockIdx.x;
    const auto c = blockIdx.y;
    const auto p = threadIdx.x;

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

    for (auto out_y = threadIdx.y; out_y < H_out; out_y += blockDim.y)
    {
        const int in_y1 = out_y * stride_h + s1.h;
        const int in_y2 = out_y * stride_h + s2.h;

        for (auto out_x = threadIdx.z; out_x < W_out; out_x += blockDim.z)
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

template <typename scalar_t>
__global__ void
full_treelogic_forward_kernel(const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> x,
                              const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,
                              torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> y1,
                              torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> y2,
                              torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> indices)
{
    const auto c = blockIdx.y * blockDim.y + threadIdx.y;
    const auto flat_index = threadIdx.x;
    if (flat_index >= indices.size(2) || c >= indices.size(1))
        return;

    auto out_y = flat_index / y2.size(3);
    auto out_x = flat_index % y2.size(3);

    const auto n = blockIdx.x;

    const scalar_t *prob1 = &weights[c][TREE1_START][1];
    const scalar_t *prob2 = &weights[c][TREE1_START + 1][1];
    const scalar_t *prob3 = &weights[c][TREE2_START][1];

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int32_t max_abs_idx = -1;

    for (int dy = 0; dy < 2; ++dy)
    {
        const int in_y = out_y * 2 + dy;
        for (int dx = 0; dx < 2; ++dx)
        {
            const int in_x = out_x * 2 + dx;

            scalar_t x1, x2, x3, x4;
            vec4_load(&x[n][c][in_y][in_x][0], x1, x2, x3, x4);

            scalar_t y1_in = prob1[0] * (x1 * x2) + prob1[1] * (x1 - x1 * x2) + prob1[2] * x1 +
                             prob1[3] * (x2 - x1 * x2) + prob1[4] * x2 + prob1[5] * (x1 + x2 - scalar_t(2) * x1 * x2) +
                             prob1[6] * (x1 + x2 - x1 * x2) + prob1[7] * (scalar_t(1) - (x1 + x2 - x1 * x2)) +
                             prob1[8] * (scalar_t(1) - (x1 + x2 - scalar_t(2) * x1 * x2)) +
                             prob1[9] * (scalar_t(1) - x2) + prob1[10] * (scalar_t(1) - x2 + x1 * x2) +
                             prob1[11] * (scalar_t(1) - x1) + prob1[12] * (scalar_t(1) - x1 + x1 * x2) +
                             prob1[13] * (scalar_t(1) - x1 * x2) + prob1[14];

            scalar_t y2_in = prob2[0] * (x3 * x4) + prob2[1] * (x3 - x3 * x4) + prob2[2] * x3 +
                             prob2[3] * (x4 - x3 * x4) + prob2[4] * x4 + prob2[5] * (x3 + x4 - scalar_t(2) * x3 * x4) +
                             prob2[6] * (x3 + x4 - x3 * x4) + prob2[7] * (scalar_t(1) - (x3 + x4 - x3 * x4)) +
                             prob2[8] * (scalar_t(1) - (x3 + x4 - scalar_t(2) * x3 * x4)) +
                             prob2[9] * (scalar_t(1) - x4) + prob2[10] * (scalar_t(1) - x4 + x3 * x4) +
                             prob2[11] * (scalar_t(1) - x3) + prob2[12] * (scalar_t(1) - x3 + x3 * x4) +
                             prob2[13] * (scalar_t(1) - x3 * x4) + prob2[14];

            vec2_store(&y1[n][c][in_y][in_x][0], y1_in, y2_in);

            scalar_t y_val = prob3[0] * (y1_in * y2_in) + prob3[1] * (y1_in - y1_in * y2_in) + prob3[2] * y1_in +
                             prob3[3] * (y2_in - y1_in * y2_in) + prob3[4] * y2_in +
                             prob3[5] * (y1_in + y2_in - scalar_t(2) * y1_in * y2_in) +
                             prob3[6] * (y1_in + y2_in - y1_in * y2_in) +
                             prob3[7] * (scalar_t(1) - (y1_in + y2_in - y1_in * y2_in)) +
                             prob3[8] * (scalar_t(1) - (y1_in + y2_in - scalar_t(2) * y1_in * y2_in)) +
                             prob3[9] * (scalar_t(1) - y2_in) + prob3[10] * (scalar_t(1) - y2_in + y1_in * y2_in) +
                             prob3[11] * (scalar_t(1) - y1_in) + prob3[12] * (scalar_t(1) - y1_in + y1_in * y2_in) +
                             prob3[13] * (scalar_t(1) - y1_in * y2_in) + prob3[14];

            if (y_val > max_val)
            {
                max_val = y_val;
                max_abs_idx = in_y * y1.size(3) + in_x;
            }
        }
    }

    y2[n][c][out_y][out_x] = max_val;
    indices[n][c][flat_index] = max_abs_idx;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
full_convlogic_cuda_forward(torch::Tensor x, torch::Tensor weights, torch::Tensor selection, int stride_h, int stride_w,
                            int pad_h, int pad_w, int kernel_h, int kernel_w)
{
    CHECK_INPUT(x);
    CHECK_INPUT(weights);
    CHECK_INPUT(selection);

    const auto N = x.size(0);
    const auto conv_H_in = x.size(2);
    const auto conv_W_in = x.size(3);
    const auto conv_C_out = weights.size(0);
    const auto conv_H_out = (conv_H_in + 2 * pad_h - kernel_h) / stride_h + 1;
    const auto conv_W_out = (conv_W_in + 2 * pad_w - kernel_w) / stride_w + 1;

    TORCH_CHECK(conv_H_out > 0 && conv_W_out > 0, "Invalid output shape: H_out=", conv_H_out, ", W_out=", conv_W_out);

    TORCH_CHECK(conv_H_out % 2 == 0 && conv_W_out % 2 == 0,
                "Output height and width after convolution must be divisible by 2 for 2x2 max pooling, but got ",
                "H_out=", conv_H_out, ", W_out=", conv_W_out);

    constexpr auto conv_p_total = 4;
    auto conv_threads_y = min(conv_H_out, int64_t(8));
    auto conv_threads_z = min(conv_W_out, int64_t(4));

    auto conv_x2 = torch::empty({N, conv_C_out, conv_H_out, conv_W_out, conv_p_total}, x.options());

    CHECK_CUDA_BLOCK_LIMIT(conv_p_total * conv_threads_y * conv_threads_z, "full_convlogic_forward_kernel");
    dim3 conv_grid(N, conv_C_out);
    dim3 conv_block(conv_p_total, conv_threads_y, conv_threads_z);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "full_convlogic_forward_kernel",
                               (
                                   [&]
                                   {
                                       full_convlogic_forward_kernel<scalar_t>
                                           <<<conv_grid, conv_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                               x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               selection.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                               conv_x2.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                                               stride_h, stride_w, pad_h, pad_w);
                                   }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto tree1_x3 = torch::empty({N, conv_C_out, conv_H_out, conv_W_out, conv_p_total / 2}, x.options());
    auto tree2_y = torch::empty({N, conv_C_out, conv_H_out / 2, conv_W_out / 2}, x.options());
    auto indices = torch::empty({N, conv_C_out, (conv_H_out / 2) * (conv_W_out / 2)},
                                torch::dtype(torch::kInt32).device(x.device()));

    const auto threads_i = indices.size(2);

    int64_t threads_c = 1;
    if (threads_i < 32)
    {
        threads_c = 32 / threads_i;
        threads_c = min(threads_c, conv_C_out);
    }

    auto blocks_c = ceil_div(conv_C_out, threads_c);

    CHECK_CUDA_BLOCK_LIMIT(threads_i * threads_c, "full_treelogic_forward_kernel");
    dim3 tree_grid(N, blocks_c);
    dim3 tree_block(threads_i, threads_c);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "full_treelogic_forward_kernel",
                               (
                                   [&]
                                   {
                                       full_treelogic_forward_kernel<scalar_t>
                                           <<<tree_grid, tree_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                               conv_x2.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                                               weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               tree1_x3.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                                               tree2_y.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>());
                                   }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return std::make_tuple(tree2_y, conv_x2, tree1_x3, indices);
}

template <typename scalar_t>
__global__ void
full_convlogic_backward_x_kernel(const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_y,
                                 const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
                                 const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,
                                 const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> selection,
                                 torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_x,
                                 torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> indices,
                                 int stride_h, int stride_w, int pad_h, int pad_w, int W_out, int indices_size,
                                 int H_in, int W_in)
{
    const auto n = blockIdx.x;
    const auto c_out = blockIdx.y * blockDim.x + threadIdx.x;
    const auto tid_i = blockIdx.z * blockDim.y + threadIdx.y;
    const auto p = threadIdx.z;
    if (tid_i >= indices_size || c_out >= indices.size(1))
        return;

    auto s1 = unpack_selection(selection[c_out][2 * p]);
    auto s2 = unpack_selection(selection[c_out][2 * p + 1]);

    s1.h -= pad_h;
    s1.w -= pad_w;
    s2.h -= pad_h;
    s2.w -= pad_w;

    const scalar_t *prob = &weights[c_out][p + CONV_START][1];

    auto flat_index = indices[n][c_out][tid_i];
    auto out_y = flat_index / W_out;
    auto out_x = flat_index % W_out;

    const int in_y1 = out_y * stride_h + s1.h;
    const int in_y2 = out_y * stride_h + s2.h;
    const int in_x1 = out_x * stride_w + s1.w;
    const int in_x2 = out_x * stride_w + s2.w;

    scalar_t grad_y_val = grad_y[n][c_out][tid_i][p];
    bool valid1 = in_y1 >= 0 && in_y1 < H_in && in_x1 >= 0 && in_x1 < W_in;
    scalar_t x1 = (valid1) ? x[n][s1.ch][in_y1][in_x1] : 0;
    bool valid2 = in_y2 >= 0 && in_y2 < H_in && in_x2 >= 0 && in_x2 < W_in;
    scalar_t x2 = (valid2) ? x[n][s2.ch][in_y2][in_x2] : 0;

    if (valid1)
    {
        const auto dy_dx1 = prob[0] * x2 + prob[1] * (scalar_t(1) - x2) + prob[2] + prob[3] * (-x2) +
                            prob[5] * (scalar_t(1) - scalar_t(2) * x2) + prob[6] * (scalar_t(1) - x2) +
                            prob[7] * (x2 - scalar_t(1)) + prob[8] * (scalar_t(2) * x2 - scalar_t(1)) + prob[10] * x2 -
                            prob[11] + prob[12] * (x2 - scalar_t(1)) + prob[13] * (-x2);
        atomicAdd(&grad_x[n][s1.ch][in_y1][in_x1], dy_dx1 * grad_y_val);
    }

    if (valid2)
    {
        const auto dy_dx2 = prob[0] * x1 + prob[1] * (-x1) + prob[3] * (scalar_t(1) - x1) + prob[4] +
                            prob[5] * (scalar_t(1) - scalar_t(2) * x1) + prob[6] * (scalar_t(1) - x1) +
                            prob[7] * (x1 - scalar_t(1)) + prob[8] * (scalar_t(2) * x1 - scalar_t(1)) - prob[9] +
                            prob[10] * (x1 - scalar_t(1)) + prob[12] * x1 + prob[13] * (-x1);
        atomicAdd(&grad_x[n][s2.ch][in_y2][in_x2], dy_dx2 * grad_y_val);
    }
}

template <typename scalar_t>
__global__ void
full_treelogic_backward_x_kernel(const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_y,
                                 const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> x_in2,
                                 const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> x_in1,
                                 const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> weights,
                                 torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_x2,
                                 torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_x1,
                                 torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> indices)
{
    const auto n = blockIdx.x;
    const auto c = blockIdx.y * blockDim.x + threadIdx.x;
    const auto tid_i = blockIdx.z * blockDim.y + threadIdx.y;
    if (tid_i >= indices.size(2) || c >= indices.size(1))
        return;

    const auto W_in = x_in1.size(3);

    const scalar_t *prob1 = &weights[c][TREE2_START][1];
    const scalar_t *prob2 = &weights[c][TREE1_START][1];
    const scalar_t *prob3 = &weights[c][TREE1_START + 1][1];

    auto flat_index = indices[n][c][tid_i];
    auto out_y = flat_index / W_in;
    auto out_x = flat_index % W_in;

    scalar_t x1, x2;
    vec2_load(&x_in2[n][c][out_y][out_x][0], x1, x2);
    scalar_t grad_y_val = grad_y[n][c][out_y / 2][out_x / 2];

    scalar_t g0 = prob1[0] * x2 + prob1[1] * (scalar_t(1) - x2) + prob1[2] + prob1[3] * (-x2) +
                  prob1[5] * (scalar_t(1) - scalar_t(2) * x2) + prob1[6] * (scalar_t(1) - x2) +
                  prob1[7] * (x2 - scalar_t(1)) + prob1[8] * (scalar_t(2) * x2 - scalar_t(1)) + prob1[10] * x2 -
                  prob1[11] + prob1[12] * (x2 - scalar_t(1)) + prob1[13] * (-x2);

    scalar_t g1 = prob1[0] * x1 + prob1[1] * (-x1) + prob1[3] * (scalar_t(1) - x1) + prob1[4] +
                  prob1[5] * (scalar_t(1) - scalar_t(2) * x1) + prob1[6] * (scalar_t(1) - x1) +
                  prob1[7] * (x1 - scalar_t(1)) + prob1[8] * (scalar_t(2) * x1 - scalar_t(1)) - prob1[9] +
                  prob1[10] * (x1 - scalar_t(1)) + prob1[12] * x1 + prob1[13] * (-x1);

    vec2_store(&grad_x2[n][c][tid_i][0], grad_y_val * g0, grad_y_val * g1);

    scalar_t x3, x4;
    vec4_load(&x_in1[n][c][out_y][out_x][0], x1, x2, x3, x4);
    scalar_t gy0, gy1;
    vec2_load(&grad_x2[n][c][tid_i][0], gy0, gy1);

    g0 = prob2[0] * x2 + prob2[1] * (scalar_t(1) - x2) + prob2[2] + prob2[3] * (-x2) +
         prob2[5] * (scalar_t(1) - scalar_t(2) * x2) + prob2[6] * (scalar_t(1) - x2) + prob2[7] * (x2 - scalar_t(1)) +
         prob2[8] * (scalar_t(2) * x2 - scalar_t(1)) + prob2[10] * x2 - prob2[11] + prob2[12] * (x2 - scalar_t(1)) +
         prob2[13] * (-x2);

    g1 = prob2[0] * x1 + prob2[1] * (-x1) + prob2[3] * (scalar_t(1) - x1) + prob2[4] +
         prob2[5] * (scalar_t(1) - scalar_t(2) * x1) + prob2[6] * (scalar_t(1) - x1) + prob2[7] * (x1 - scalar_t(1)) +
         prob2[8] * (scalar_t(2) * x1 - scalar_t(1)) - prob2[9] + prob2[10] * (x1 - scalar_t(1)) + prob2[12] * x1 +
         prob2[13] * (-x1);

    scalar_t g2 = prob3[0] * x4 + prob3[1] * (scalar_t(1) - x4) + prob3[2] + prob3[3] * (-x4) +
                  prob3[5] * (scalar_t(1) - scalar_t(2) * x4) + prob3[6] * (scalar_t(1) - x4) +
                  prob3[7] * (x4 - scalar_t(1)) + prob3[8] * (scalar_t(2) * x4 - scalar_t(1)) + prob3[10] * x4 -
                  prob3[11] + prob3[12] * (x4 - scalar_t(1)) + prob3[13] * (-x4);

    scalar_t g3 = prob3[0] * x3 + prob3[1] * (-x3) + prob3[3] * (scalar_t(1) - x3) + prob3[4] +
                  prob3[5] * (scalar_t(1) - scalar_t(2) * x3) + prob3[6] * (scalar_t(1) - x3) +
                  prob3[7] * (x3 - scalar_t(1)) + prob3[8] * (scalar_t(2) * x3 - scalar_t(1)) - prob3[9] +
                  prob3[10] * (x3 - scalar_t(1)) + prob3[12] * x3 + prob3[13] * (-x3);

    vec4_store(&grad_x1[n][c][tid_i][0], gy0 * g0, gy0 * g1, gy1 * g2, gy1 * g3);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
full_convlogic_cuda_backward_x(torch::Tensor grad_y, torch::Tensor conv_x, torch::Tensor tree1_x, torch::Tensor tree2_x,
                               torch::Tensor weights, torch::Tensor indices, torch::Tensor selection, int stride_h,
                               int stride_w, int pad_h, int pad_w)
{
    CHECK_INPUT(grad_y);
    CHECK_INPUT(conv_x);
    CHECK_INPUT(tree1_x);
    CHECK_INPUT(tree2_x);
    CHECK_INPUT(weights);
    CHECK_INPUT(indices);
    CHECK_INPUT(selection);

    const auto N = tree2_x.size(0);
    const auto C_out = tree2_x.size(1);

    const auto tree2_Xout = 1;
    const auto tree1_Xout = 2;

    auto tree1_grad_x = torch::empty({N, C_out, indices.size(2), 2 * tree1_Xout}, tree1_x.options());
    auto tree2_grad_x = torch::empty({N, C_out, indices.size(2), 2 * tree2_Xout}, tree2_x.options());

    int64_t threads_i = indices.size(2);
    int64_t threads_c = 1;

    // Boost thread count in c-dimension if threads_i is small
    if (threads_i < 32)
    {
        threads_c = 32 / threads_i;
        threads_c = min(threads_c, C_out);
    }

    auto blocks_i = ceil_div(indices.size(2), threads_i);
    auto blocks_c = ceil_div(C_out, threads_c);

    CHECK_CUDA_BLOCK_LIMIT(threads_c * threads_i, "full_treelogic_backward_x_kernel");
    dim3 grid(N, blocks_c, blocks_i);
    dim3 block(threads_c, threads_i);

    AT_DISPATCH_FLOATING_TYPES(tree2_x.scalar_type(), "full_treelogic_backward_x_kernel",
                               (
                                   [&]
                                   {
                                       full_treelogic_backward_x_kernel<scalar_t>
                                           <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                               grad_y.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               tree2_x.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                                               tree1_x.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                                               weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               tree2_grad_x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               tree1_grad_x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>());
                                   }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto conv_grad_x = torch::zeros_like(conv_x);

    auto indices_size = indices.size(2);
    threads_i = indices_size;
    threads_i = min(threads_i, int64_t(MAX_THREADS_PER_BLOCK / 4));

    if ((threads_i * 4) < 32)
    {
        threads_c = 32 / (threads_i * 4);
        threads_c = min(threads_c, C_out);
    }

    blocks_c = ceil_div(C_out, threads_c);
    blocks_i = ceil_div(indices_size, threads_i);

    CHECK_CUDA_BLOCK_LIMIT(threads_c * threads_i * 4, "full_convlogic_backward_x_kernel");
    dim3 grid_conv(N, blocks_c, blocks_i);
    dim3 block_conv(threads_c, threads_i, 4);

    AT_DISPATCH_FLOATING_TYPES(conv_x.scalar_type(), "full_convlogic_backward_x_kernel",
                               (
                                   [&]
                                   {
                                       full_convlogic_backward_x_kernel<scalar_t>
                                           <<<grid_conv, block_conv, 0, at::cuda::getCurrentCUDAStream()>>>(
                                               tree1_grad_x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               conv_x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               weights.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                               selection.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                               conv_grad_x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                                               indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                                               stride_h, stride_w, pad_h, pad_w, tree1_x.size(3), indices_size,
                                               conv_x.size(2), conv_x.size(3));
                                   }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return std::make_tuple(conv_grad_x, tree1_grad_x, tree2_grad_x);
}

template <typename scalar_t>
__global__ void full_treelogic_backward_weight_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_x3,
    const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> x_in2,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_y,
    const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> x_in3,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_w_4,
    torch::PackedTensorAccessor64<int32_t, 3, torch::RestrictPtrTraits> indices)
{
    const auto n = blockIdx.x;
    const auto c = blockIdx.y;
    const auto p = threadIdx.x;
    const auto tid_i = threadIdx.y;

    auto W_out = x_in3.size(3);

    extern __shared__ char smem_raw[];
    scalar_t *smem = reinterpret_cast<scalar_t *>(smem_raw);

    scalar_t *shared_ptr = smem + (p * blockDim.y + tid_i) * 8;

    scalar_t gate_vals[8] = {0};

    for (auto i = tid_i; i < indices.size(2); i += blockDim.y)
    {
        auto flat_index = indices[n][c][i];
        auto out_y = flat_index / W_out;
        auto out_x = flat_index % W_out;

        scalar_t gy_val = grad_x3[n][c][i][p];
        scalar_t x1, x2;
        vec2_load(&x_in2[n][c][out_y][out_x][2 * p], x1, x2);

        gate_vals[0] += gy_val * (x1 * x2);
        gate_vals[1] += gy_val * x1;
        gate_vals[2] += gy_val * x2;
        gate_vals[3] += gy_val;

        if (p == 0)
        {
            gy_val = grad_y[n][c][out_y / 2][out_x / 2];
            vec2_load(&x_in3[n][c][out_y][out_x][0], x1, x2);

            gate_vals[4] += gy_val * (x1 * x2);
            gate_vals[5] += gy_val * x1;
            gate_vals[6] += gy_val * x2;
            gate_vals[7] += gy_val;
        }
    }

    for (auto g = 0; g < 8; ++g)
    {
        shared_ptr[g] = gate_vals[g];
    }

    __syncthreads();

    if (tid_i == 0)
    {
        for (auto g = 0; g < 4; ++g)
        {
            scalar_t acc = 0;
            for (auto z = 0; z < blockDim.y; ++z)
            {
                acc += smem[(p * blockDim.y + z) * 8 + g];
            }
            grad_w_4[c][TREE1_START + p][g][n] = acc;
        }
    }
    if (p == 0 && tid_i == 0)
    {
        for (auto g = 0; g < 4; ++g)
        {
            scalar_t acc = 0;
            for (auto z = 0; z < blockDim.y; ++z)
            {
                acc += smem[z * 8 + g + 4];
            }
            grad_w_4[c][TREE2_START][g][n] = acc;
        }
    }
}

template <typename scalar_t>
__global__ void full_convlogic_backward_weight_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_y,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> selection,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_w_4,
    torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> indices, int stride_h, int stride_w, int pad_h,
    int pad_w, int W_out)
{
    const auto n = blockIdx.x;
    const auto c_out = blockIdx.y;
    const auto p = threadIdx.x;
    const auto tid_y = threadIdx.y;

    extern __shared__ char smem_raw[];
    scalar_t *smem = reinterpret_cast<scalar_t *>(smem_raw);

    scalar_t *shared_ptr = smem + (p * blockDim.y + tid_y) * 4;

    // Init local accumulators
    scalar_t gate_vals[4] = {0};

    auto s1 = unpack_selection(selection[c_out][2 * p]);
    auto s2 = unpack_selection(selection[c_out][2 * p + 1]);

    for (auto i = tid_y; i < indices.size(2); i += blockDim.y)
    {
        auto flat_index = indices[n][c_out][i];
        auto out_y = flat_index / W_out;
        auto out_x = flat_index % W_out;

        auto in_y1 = out_y * stride_h + s1.h;
        auto in_y2 = out_y * stride_h + s2.h;
        auto in_x1 = out_x * stride_w + s1.w;
        auto in_x2 = out_x * stride_w + s2.w;

        scalar_t gy = grad_y[n][c_out][i][p];

        scalar_t x1 = x[n][s1.ch][in_y1][in_x1];
        scalar_t x2 = x[n][s2.ch][in_y2][in_x2];

        gate_vals[0] += gy * (x1 * x2);
        gate_vals[1] += gy * x1;
        gate_vals[2] += gy * x2;
        gate_vals[3] += gy;
    }

    // Write partials to shared memory
    for (auto g = 0; g < 4; ++g)
    {
        shared_ptr[g] = gate_vals[g];
    }

    __syncthreads();

    // First thread per (p, gate) reduces across z
    if (tid_y == 0)
    {
        for (auto g = 0; g < 4; ++g)
        {
            scalar_t acc = 0;
            for (auto z = 0; z < blockDim.y; ++z)
            {
                acc += smem[(p * blockDim.y + z) * 4 + g];
            }
            grad_w_4[c_out][p][g][n] = acc;
        }
    }
}

torch::Tensor full_convlogic_cuda_backward_weight(torch::Tensor grad_x2, torch::Tensor grad_x3, torch::Tensor grad_y,
                                                  torch::Tensor x1, torch::Tensor x2, torch::Tensor x3,
                                                  torch::Tensor indices, torch::Tensor selection, int stride_h,
                                                  int stride_w, int pad_h, int pad_w)
{
    CHECK_INPUT(grad_y);
    CHECK_INPUT(x1);
    CHECK_INPUT(x2);
    CHECK_INPUT(x3);
    CHECK_INPUT(indices);
    CHECK_INPUT(selection);

    // Convolution backward weight
    const auto N = grad_y.size(0);
    const auto C_out = grad_y.size(1);

    constexpr auto conv_p_total = 4;
    constexpr auto conv_threads_y = 8;

    auto conv_x1_padded = torch::constant_pad_nd(x1, {pad_w, pad_w, pad_h, pad_h}, 0);
    auto grad_w_4 = torch::empty({C_out, 7, 4, N}, torch::dtype(x1.dtype()).device(x1.device()));

    CHECK_CUDA_BLOCK_LIMIT(conv_p_total * conv_threads_y, "full_convlogic_backward_weight_kernel");
    dim3 conv_grid(N, C_out);
    dim3 conv_block(conv_p_total, conv_threads_y);

    AT_DISPATCH_FLOATING_TYPES(
        x1.scalar_type(), "full_convlogic_backward_weight_kernel",
        (
            [&]
            {
                size_t conv_shared_mem = conv_p_total * 4 * conv_threads_y * sizeof(scalar_t);
                CHECK_SHARED_MEMORY(conv_shared_mem);

                full_convlogic_backward_weight_kernel<scalar_t>
                    <<<conv_grid, conv_block, conv_shared_mem, at::cuda::getCurrentCUDAStream()>>>(
                        grad_x2.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                        conv_x1_padded.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                        selection.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                        grad_w_4.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                        indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(), stride_h, stride_w, pad_h,
                        pad_w, x2.size(3));
            }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    const auto tree1_Xout = 2;
    auto threads_i = indices.size(2);
    threads_i = min(threads_i, int64_t(MAX_THREADS_PER_BLOCK / 2));

    CHECK_CUDA_BLOCK_LIMIT(tree1_Xout * threads_i, "full_treelogic_backward_weight_kernel");
    dim3 tree_grid(N, C_out);
    dim3 tree_block(tree1_Xout, threads_i);

    AT_DISPATCH_FLOATING_TYPES(
        x2.scalar_type(), "full_treelogic_backward_weight_kernel",
        (
            [&]
            {
                size_t tree_shared_mem = tree1_Xout * threads_i * 8 * sizeof(scalar_t);
                CHECK_SHARED_MEMORY(tree_shared_mem);

                full_treelogic_backward_weight_kernel<scalar_t>
                    <<<tree_grid, tree_block, tree_shared_mem, at::cuda::getCurrentCUDAStream()>>>(
                        grad_x3.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                        x2.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                        grad_y.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                        x3.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                        grad_w_4.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                        indices.packed_accessor64<int32_t, 3, torch::RestrictPtrTraits>());
            }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto coefs_T = torch::tensor({{0, 1, -1, 0, -1, 0, -2, -1, 1, 2, 0, 1, 0, 1, -1, 0},
                                  {0, 0, 1, 1, 0, 0, 1, 1, -1, -1, 0, 0, -1, -1, 0, 0},
                                  {0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0},
                                  {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}},
                                 torch::dtype(x1.dtype()).device(x1.device()));

    auto grad_w_components = grad_w_4.sum(3);             // [C_out, 7, 4]
    auto grad_weight = grad_w_components.matmul(coefs_T); // [C_out, 7, 16]
    return grad_weight;
}
