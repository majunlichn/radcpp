#pragma once

#include <rad/Core/Platform.h>
#include <rad/Core/Integer.h>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <limits>

#include <boost/math/special_functions/relative_difference.hpp>

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

// E = fabs((a - b) / min(a,b));
template <typename T, typename U>
auto relative_difference(T a, U b)
{
    return boost::math::relative_difference(a, b);
}

// returns relative_difference(a, b) / eps where eps is the machine epsilon for the result type.
template <typename T, typename U>
auto epsilon_difference(T a, U b)
{
    return boost::math::epsilon_difference(a, b);
}

// https://entity-toolkit.github.io/wiki/useful/float-comparison/#the-simpler-way
template <class T>
auto AlmostEqual(T a, T b, T eps = std::numeric_limits<T>::epsilon()) -> bool
{
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type!");
    return (a == b) ||
        (std::fabs(a - b) <= std::min(std::fabs(a), std::fabs(b)) * eps);
}

float Normalize(float value, float min, float max);
uint8_t QuantizeUnorm8(float normalized);
uint16_t QuantizeUnorm16(float normalized);
uint32_t QuantizeUnorm32(double normalized);
float DequantizeUnorm8(uint8_t quantized);
float DequantizeUnorm16(uint16_t quantized);
float DequantizeUnorm32(uint32_t quantized);

} // namespace rad
