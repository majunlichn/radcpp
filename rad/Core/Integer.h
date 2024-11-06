#pragma once

#include <rad/Core/Platform.h>
#include <cassert>
#include <cstdint>
#include <bit>
// has_single_bit: checks if a number is an integral power of 2.
// bit_ceil: finds the smallest integral power of two not less than the given value.
// bit_floor: finds the largest integral power of two not greater than the given value.
// bit_width: finds the smallest number of bits needed to represent the given value.
// rotl: computes the result of bitwise left-rotation.
// rotr: computes the result of bitwise right-rotation.
// countl_zero: counts the number of consecutive ​0​ bits, starting from the most significant bit.
// countl_one: counts the number of consecutive 1 bits, starting from the most significant bit.
// countr_zero: counts the number of consecutive ​0​ bits, starting from the least significant bit.
// countr_one: counts the number of consecutive 1 bits, starting from the least significant bit.
// popcount: counts the number of 1 bits in an unsigned integer.
#include <concepts>

namespace rad
{

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

template<std::unsigned_integral T>
constexpr uint32_t CountBits(T x)
{
    return std::popcount(x);
}

// Count the number of bits set in an unsigned integer (popcount).
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
uint32_t RoundUpToNextPow2(uint32_t x);
uint64_t RoundUpToNextPow2(uint64_t x);
uint32_t RoundUpToPow2(uint32_t x);
uint64_t RoundUpToPow2(uint64_t x);
#endif

} // namespace rad
