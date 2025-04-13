#pragma once

#include <rad/Core/Platform.h>
#include <rad/Core/Integer.h>
#include <cfenv>
#include <cfloat>
#include <cmath>

#include <concepts>
#include <limits>

#include <boost/math/special_functions/next.hpp>
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

// Finding the next representable value in a specific direction.
using boost::math::nextafter;
// Finding the next greater representable value.
using boost::math::float_next;
// Finding the next smaller representable value.
using boost::math::float_prior;
// Calculating the representation distance between two floating-point values (ULP).
using boost::math::float_distance;
// Advancing a floating-point value by a specific representation distance (ULP).
using boost::math::float_advance;
// E = fabs((a - b) / min(a,b))
using boost::math::relative_difference;
// a convenience function that returns relative_difference(a, b) / epsilon.
using boost::math::epsilon_difference;

// https://entity-toolkit.github.io/wiki/useful/float-comparison/#the-simpler-way
template <std::floating_point T>
auto AlmostEqual(T a, T b, T eps = std::numeric_limits<T>::epsilon()) -> bool
{
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

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
uint32_t fp16_ieee_to_fp32_bits(uint16_t h);

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
float fp16_ieee_to_fp32_value(uint16_t h);

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 16-bit floating-point number in IEEE half-precision format, in bit
 * representation.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
uint16_t fp16_ieee_from_fp32_value(float f);


// The bfloat16 format was developed by Google Brain, an artificial intelligence research group at Google.
// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
// sign: 1 bit; exponent: 8 bits; fraction: 7 bits;
// precision: between two and three decimal digits.
// range: up to about 3.4 × 10^38.
// max: 0x7f7f = 0 11111110 1111111 = (28 - 1) * 2^-7 * 2^127; about 3.38953139 * 10^38 < FLT_MAX
// min: 0x0080 = 0 00000001 0000000 = 2^-126; about 1.175494351 * 10^-38
// min (subnormal): 2^-126-7=2-133; about 9.2 * 10^-41
#define BF16_MIN 0x0080
#define BF16_MAX 0x7f7f
// 0000 = 0 00000000 0000000 = 0
// 8000 = 1 00000000 0000000 = −0
// 7f80 = 0 11111111 0000000 = infinity
// ff80 = 1 11111111 0000000 = −infinity
// 4049 = 0 10000000 1001001 = 3.140625; // pi
// 3eab = 0 01111101 0101011 = 0.333984375 // 1/3
// NaN: not all significand bits zero.
// ffc1 = x 11111111 1000001 => qNaN
// ff81 = x 11111111 0000001 => sNaN

uint16_t bf16_from_fp32_round_to_zero(float x);
uint16_t bf16_from_fp32_round_to_nearest_even(float x);
float bf16_to_fp32(uint16_t x);

float fp8e4m3fn_to_fp32_value(uint8_t input);
uint8_t fp8e4m3fn_from_fp32_value(float f);
float fp8e5m2_to_fp32_value(uint8_t input);
uint8_t fp8e5m2_from_fp32_value(float f);

} // namespace rad
