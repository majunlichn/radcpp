#pragma once

#include <rad/Core/Platform.h>
#include <rad/Core/Integer.h>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <limits>

namespace rad
{

// https://github.com/pytorch/pytorch/blob/main/c10/util/floating_point_utils.h
inline float fp32_from_bits(uint32_t w)
{
#if defined(__OPENCL_VERSION__)
    return as_float(w);
#elif defined(__CUDA_ARCH__)
    return __uint_as_float((unsigned int)w);
#elif defined(__INTEL_COMPILER)
    return _castu32_f32(w);
#else
    union {
        uint32_t as_bits;
        float as_value;
    } fp32 = { w };
    return fp32.as_value;
#endif
}

// https://github.com/pytorch/pytorch/blob/main/c10/util/floating_point_utils.h
inline uint32_t fp32_to_bits(float f)
{
#if defined(__OPENCL_VERSION__)
    return as_uint(f);
#elif defined(__CUDA_ARCH__)
    return (uint32_t)__float_as_uint(f);
#elif defined(__INTEL_COMPILER)
    return _castf32_u32(f);
#else
    union {
        float as_value;
        uint32_t as_bits;
    } fp32 = { f };
    return fp32.as_bits;
#endif
}

float Normalize(float value, float min, float max);
uint8_t QuantizeUnorm8(float normalized);
uint16_t QuantizeUnorm16(float normalized);
uint32_t QuantizeUnorm32(double normalized);
float DequantizeUnorm8(uint8_t quantized);
float DequantizeUnorm16(uint16_t quantized);
float DequantizeUnorm32(uint32_t quantized);

} // namespace rad
