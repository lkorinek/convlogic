#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <type_traits>
#include <cmath>

// Tensor checks
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// Kernel config checks
#define CHECK_CUDA_BLOCK_LIMIT(requested_threads, label)                                         \
    {                                                                                              \
        int __device_id__;                                                                         \
        cudaGetDevice(&__device_id__);                                                             \
        int __max_threads__;                                                                       \
        cudaDeviceGetAttribute(&__max_threads__, cudaDevAttrMaxThreadsPerBlock, __device_id__);    \
        TORCH_CHECK((requested_threads) > 0 && (requested_threads) <= __max_threads__,             \
                    "Block size for " label " exceeds device limit: ",                             \
                    (requested_threads), " > ", __max_threads__);                                  \
    }

#define CHECK_SHARED_MEMORY(required_bytes)                                                       \
    {                                                                                             \
        int __device_id__;                                                                        \
        cudaGetDevice(&__device_id__);                                                            \
        int __max_shared__;                                                                       \
        cudaDeviceGetAttribute(&__max_shared__, cudaDevAttrMaxSharedMemoryPerBlock, __device_id__); \
        TORCH_CHECK((size_t)(required_bytes) <= (size_t)__max_shared__,                           \
                    "Requested shared memory (", (required_bytes), " bytes) exceeds max (",       \
                    __max_shared__, " bytes).");                                                  \
    }

// CUDA error check
// Adapted from
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
// Used to detect and report CUDA errors during execution.
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Selection helper
struct SelectionTriple
{
    int ch, h, w;
};

/**
 * @brief Unpacks a 32-bit packed selection value into a SelectionTriple.
 *
 * bits 31..16 → channel
 * bits 15..8  → row (h)
 * bits 7..0   → col (w)
 */
__device__ __forceinline__ SelectionTriple unpack_selection(int32_t packed)
{
    return SelectionTriple{
        .ch = (packed >> 16) & 0xFFFF,
        .h  = (packed >> 8) & 0xFF,
        .w  = packed & 0xFF
    };
}

// Math utility used for calculating the number of blocks needed for a given number of threads and threads per block.
template <typename T> T ceil_div(const T x, const T y)
{
    return x / y + !!(x % y);
}

// Atomic float/half/double wrapper
template <typename T> struct AtomicFPOp;

template <> struct AtomicFPOp<at::Half>
{
    template <typename func_t>
    inline __device__ at::Half operator()(at::Half *address, at::Half val, const func_t &func)
    {
        unsigned int *address_as_ui = (unsigned int *)((char *)address - ((size_t)address & 2));
        unsigned int old = *address_as_ui;
        unsigned int assumed;

        at::Half hsum;
        do
        {
            assumed = old;
            hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
            hsum = func(hsum, val);
            old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
            old = atomicCAS(address_as_ui, assumed, old);
        } while (assumed != old);
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        return hsum;
    }
};

static inline __device__ at::Half gpuAtomicAdd(at::Half *address, at::Half val)
{
#if defined(USE_ROCM) ||                                                                                               \
    ((defined(CUDA_VERSION) && CUDA_VERSION < 10000) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))

    unsigned int *aligned = (unsigned int *)((size_t)address - ((size_t)address & 2));
    unsigned int old = *aligned;
    unsigned int assumed;
    do
    {
        assumed = old;
        unsigned short old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
        __half sum = c10::Half(__ushort_as_half(old_as_us)) + c10::Half(__float2half((float)val));
        unsigned short sum_as_us = __half_as_ushort(sum);
        unsigned int sum_as_ui =
            (size_t)address & 2 ? (sum_as_us << 16) | (old & 0xffff) : (old & 0xffff0000) | sum_as_us;
        old = atomicCAS(aligned, assumed, sum_as_ui);
    } while (assumed != old);
    unsigned short old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
    return c10::Half((__half_raw)__ushort_as_half(old_as_us));
#else
    return atomicAdd(reinterpret_cast<__half *>(address), val);
#endif
}

static inline __device__ float gpuAtomicAdd(float *address, float val)
{
    return atomicAdd(address, val);
}

static inline __device__ double gpuAtomicAdd(double *address, double val)
{
    return atomicAdd(address, val);
}

// | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 |
// |----|----------------------|-------|-------|-------|-------|
// | 0  | 0                    | 0     | 0     | 0     | 0     |
// | 1  | A and B              | 0     | 0     | 0     | 1     |
// | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
// | 3  | A                    | 0     | 0     | 1     | 1     |
// | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
// | 5  | B                    | 0     | 1     | 0     | 1     |
// | 6  | A xor B              | 0     | 1     | 1     | 0     |
// | 7  | A or B               | 0     | 1     | 1     | 1     |
// | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
// | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
// | 10 | not(B)               | 1     | 0     | 1     | 0     |
// | 11 | B implies A          | 1     | 0     | 1     | 1     |
// | 12 | not(A)               | 1     | 1     | 0     | 0     |
// | 13 | A implies B          | 1     | 1     | 0     | 1     |
// | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
// | 15 | 1                    | 1     | 1     | 1     | 1     |

template <typename T> __device__ __forceinline__ T bin_op_eval(const T a_, const T b_, const int op_idx)
{
    switch (op_idx)
    {
    case 0:
        return static_cast<T>(0);
    case 1:
        return a_ & b_;
    case 2:
        return a_ & ~b_;
    case 3:
        return a_;
    case 4:
        return b_ & ~a_;
    case 5:
        return b_;
    case 6:
        return a_ ^ b_;
    case 7:
        return a_ | b_;
    case 8:
        return ~(a_ | b_);
    case 9:
        return ~(a_ ^ b_);
    case 10:
        return ~b_;
    case 11:
        return ~b_ | a_;
    case 12:
        return ~a_;
    case 13:
        return ~a_ | b_;
    case 14:
        return ~(a_ & b_);
    default:
        return ~static_cast<T>(0);
    }
}
