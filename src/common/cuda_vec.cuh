#pragma once
#include <type_traits>

template <typename scalar_t> __device__ inline void vec2_load(const scalar_t *base_ptr, scalar_t &x1, scalar_t &x2)
{
    if constexpr (std::is_same<scalar_t, float>::value)
    {
        float2 v = *reinterpret_cast<const float2 *>(base_ptr);
        x1 = v.x;
        x2 = v.y;
    }
    else if constexpr (std::is_same<scalar_t, double>::value)
    {
        double2 v = *reinterpret_cast<const double2 *>(base_ptr);
        x1 = v.x;
        x2 = v.y;
    }
}

template <typename scalar_t>
__device__ inline void vec4_load(const scalar_t *base_ptr, scalar_t &x1, scalar_t &x2, scalar_t &x3, scalar_t &x4)
{
    if constexpr (std::is_same<scalar_t, float>::value)
    {
        float4 v = *reinterpret_cast<const float4 *>(base_ptr);
        x1 = v.x;
        x2 = v.y;
        x3 = v.z;
        x4 = v.w;
    }
    else if constexpr (std::is_same<scalar_t, double>::value)
    {
        double4 v = *reinterpret_cast<const double4 *>(base_ptr);
        x1 = v.x;
        x2 = v.y;
        x3 = v.z;
        x4 = v.w;
    }
}

template <typename scalar_t>
__device__ inline void vec4_store(scalar_t *base_ptr, scalar_t x1, scalar_t x2, scalar_t x3, scalar_t x4)
{
    if constexpr (std::is_same<scalar_t, float>::value)
    {
        float4 v = {x1, x2, x3, x4};
        *reinterpret_cast<float4 *>(base_ptr) = v;
    }
    else if constexpr (std::is_same<scalar_t, double>::value)
    {
        double4 v = {x1, x2, x3, x4};
        *reinterpret_cast<double4 *>(base_ptr) = v;
    }
}

template <typename scalar_t> __device__ inline void vec2_store(scalar_t *base_ptr, scalar_t x1, scalar_t x2)
{
    if constexpr (std::is_same<scalar_t, float>::value)
    {
        float2 v = {x1, x2};
        *reinterpret_cast<float2 *>(base_ptr) = v;
    }
    else if constexpr (std::is_same<scalar_t, double>::value)
    {
        double2 v = {x1, x2};
        *reinterpret_cast<double2 *>(base_ptr) = v;
    }
}
