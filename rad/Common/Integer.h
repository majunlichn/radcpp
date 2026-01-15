#pragma once

#include <rad/Config.h>
#include <rad/Common/Platform.h>

#include <cassert>
#include <cstdint>

#include <bit>
#include <concepts>

namespace rad
{

using Sint8 = int8_t;
using Sint16 = int16_t;
using Sint32 = int32_t;
using Sint64 = int64_t;
using Uint8 = uint8_t;
using Uint16 = uint16_t;
using Uint32 = uint32_t;
using Uint64 = uint64_t;

template <class T>
constexpr bool is_integer_v = std::is_integral_v<T>;

template <class T>
constexpr bool is_signed_integer_v = std::is_integral_v<T> && std::is_signed_v<T>;

template <class T>
constexpr bool is_unsigned_integer_v = std::is_integral_v<T> && std::is_unsigned_v<T>;

template<std::integral T>
constexpr bool HasBits(T mask, T bits) noexcept
{
    return ((mask & bits) == bits);
}

template<std::integral T>
constexpr bool HasNoBits(T mask, T bits) noexcept
{
    return ((mask & bits) == 0);
}

template<std::integral T>
constexpr bool HasAnyBits(T mask, T bits) noexcept
{
    return ((mask & bits) != 0);
}

// Search the mask data from most significant bit (MSB) to least significant bit (LSB) for a set bit (1).
// The mask input must be nonzero or the index returned is undefined.
uint32_t BitScanReverse32(uint32_t mask);
uint32_t BitScanReverse64(uint64_t mask);

uint32_t CountBits32(uint32_t x);
uint32_t CountBits64(uint64_t x);

inline uint32_t ReverseBits32(uint32_t n)
{
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    return n;
}

inline uint64_t ReverseBits64(uint64_t n)
{
    uint64_t n0 = ReverseBits32((uint32_t)n);
    uint64_t n1 = ReverseBits32((uint32_t)(n >> 32));
    return (n0 << 32) | n1;
}

template<std::unsigned_integral T>
constexpr T RoundUpToMultiple(T value, T alignment)
{
    assert(alignment >= 1);
    assert((value + (alignment - 1)) >= value); // check overflow
    return (((value + (alignment - 1)) / alignment) * alignment);
}

template <std::unsigned_integral T>
constexpr T RoundDownToMultiple(T value, T alignment)
{
    assert(alignment >= 1);
    return ((value / alignment) * alignment);
}

template<std::unsigned_integral T>
constexpr bool IsPow2(T x)
{
#if defined(__cpp_lib_int_pow2)
    return std::has_single_bit(x);
#else // fallback
    return (x > 0) && ((x & (x - 1)) == 0);
#endif
}

template <std::unsigned_integral T>
T Pow2AlignUp(T value, T alignment)
{
    assert(IsPow2(alignment));
    assert((value + (alignment - 1)) >= value);
    return ((value + (alignment - 1)) & ~(alignment - 1));
}

template <std::unsigned_integral T>
T Pow2AlignDown(T value, T alignment)
{
    assert(IsPow2(alignment));
    return (value & ~(alignment - 1));
}

#if defined(__cpp_lib_int_pow2)

template<std::unsigned_integral T>
constexpr T RoundUpToNextPow2(T x)
{
    return T(1) << std::bit_width(x);
}

template<std::unsigned_integral T>
constexpr T RoundUpToPow2(T x)
{
    return std::bit_ceil(x);
}

template<std::unsigned_integral T>
constexpr T RoundDownToPow2(T x)
{
    return std::bit_floor(x);
}

#else // fallback

inline uint32_t RoundUpToNextPow2(uint32_t x)
{
    assert(x < 0x80000000);
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

inline uint64_t RoundUpToNextPow2(uint64_t x)
{
    assert(x < 0x8000000000000000);
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x |= x >> 64;
    return x + 1;
}

inline uint32_t RoundUpToPow2(uint32_t x)
{
    assert(x <= 0x80000000);
    if (x > 1)
    {
        return RoundUpToNextPow2(x - 1);
    }
    else
    {
        return 1;
    }
}

inline uint64_t RoundUpToPow2(uint64_t x)
{
    assert(x <= 0x8000000000000000);
    if (x > 1)
    {
        return RoundUpToNextPow2(x - 1);
    }
    else
    {
        return 1;
    }
}

#endif

template <std::integral T>
constexpr T DivRoundUp(T a, T b)
{
    assert(b > 0);
    return (a + b - 1) / b;
}

} // namespace rad
