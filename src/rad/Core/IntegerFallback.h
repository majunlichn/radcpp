#pragma once

#include <rad/Core/Integer.h>

// Portable implementations of selected integer operations. Prefer the primary
// functions in Integer.h unless you specifically need fallback behavior.
namespace rad::fallback
{

template <UnsignedInteger T>
[[nodiscard]] constexpr int BitScanReverse(T value) noexcept
{
    assert(value != 0);

    if constexpr (BitCount<T> == 64)
    {
        int index = 0;
        if (value & T{0xFFFFFFFF00000000ULL})
        {
            value = static_cast<T>(value >> 32);
            index += 32;
        }
        if (value & T{0xFFFF0000})
        {
            value = static_cast<T>(value >> 16);
            index += 16;
        }
        if (value & T{0xFF00})
        {
            value = static_cast<T>(value >> 8);
            index += 8;
        }
        if (value & T{0xF0})
        {
            value = static_cast<T>(value >> 4);
            index += 4;
        }
        if (value & T{0xC})
        {
            value = static_cast<T>(value >> 2);
            index += 2;
        }
        if (value & T{0x2})
        {
            index += 1;
        }
        return index;
    }
    else if constexpr (BitCount<T> == 32)
    {
        int index = 0;
        if (value & T{0xFFFF0000})
        {
            value = static_cast<T>(value >> 16);
            index += 16;
        }
        if (value & T{0xFF00})
        {
            value = static_cast<T>(value >> 8);
            index += 8;
        }
        if (value & T{0xF0})
        {
            value = static_cast<T>(value >> 4);
            index += 4;
        }
        if (value & T{0xC})
        {
            value = static_cast<T>(value >> 2);
            index += 2;
        }
        if (value & T{0x2})
        {
            index += 1;
        }
        return index;
    }
    else
    {
        int index = 0;
        while ((value = static_cast<T>(value >> 1)) != 0)
        {
            ++index;
        }
        return index;
    }
}

template <UnsignedInteger T>
[[nodiscard]] constexpr int CountSetBits(T value) noexcept
{
    if constexpr (BitCount<T> == 64)
    {
        auto x = static_cast<std::uint64_t>(value);
        x -= (x >> 1) & 0x5555555555555555ULL;
        x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
        return static_cast<int>((x * 0x0101010101010101ULL) >> 56);
    }
    else if constexpr (BitCount<T> == 32)
    {
        auto x = static_cast<std::uint32_t>(value);
        x = x - ((x >> 1) & 0x55555555U);
        x = (x & 0x33333333U) + ((x >> 2) & 0x33333333U);
        x = (((x + (x >> 4)) & 0x0F0F0F0FU) * 0x01010101U) >> 24;
        return static_cast<int>(x);
    }
    else
    {
        int count = 0;
        while (value != 0)
        {
            value = static_cast<T>(value & static_cast<T>(value - 1));
            ++count;
        }
        return count;
    }
}

template <UnsignedInteger T>
[[nodiscard]] constexpr T ReverseBits(T value) noexcept
{
    if constexpr (BitCount<T> == 64)
    {
        const auto low = static_cast<std::uint64_t>(ReverseBits(static_cast<std::uint32_t>(value)));
        const auto high =
            static_cast<std::uint64_t>(ReverseBits(static_cast<std::uint32_t>(value >> 32)));
        return static_cast<T>((low << 32) | high);
    }
    else if constexpr (BitCount<T> == 32)
    {
        auto x = static_cast<std::uint32_t>(value);
        x = (x << 16) | (x >> 16);
        x = ((x & 0x00FF00FFU) << 8) | ((x & 0xFF00FF00U) >> 8);
        x = ((x & 0x0F0F0F0FU) << 4) | ((x & 0xF0F0F0F0U) >> 4);
        x = ((x & 0x33333333U) << 2) | ((x & 0xCCCCCCCCU) >> 2);
        x = ((x & 0x55555555U) << 1) | ((x & 0xAAAAAAAAU) >> 1);
        return static_cast<T>(x);
    }
    else
    {
        T result = 0;
        for (std::size_t bit = 0; bit < BitCount<T>; ++bit)
        {
            result = static_cast<T>((result << 1) | (value & T{1}));
            value = static_cast<T>(value >> 1);
        }
        return result;
    }
}

template <UnsignedInteger T>
[[nodiscard]] constexpr T RoundUpToNextPowerOfTwo(T value) noexcept
{
    assert(value < (T{1} << (BitCount<T> - 1)));

    for (std::size_t shift = 1; shift < BitCount<T>; shift <<= 1)
    {
        value = static_cast<T>(value | (value >> shift));
    }

    return static_cast<T>(value + 1);
}

template <UnsignedInteger T>
[[nodiscard]] constexpr T RoundUpToPowerOfTwo(T value) noexcept
{
    assert(value <= (T{1} << (BitCount<T> - 1)));
    return value <= 1 ? T{1} : RoundUpToNextPowerOfTwo(static_cast<T>(value - 1));
}

template <UnsignedInteger T>
[[nodiscard]] constexpr T RoundDownToPowerOfTwo(T value) noexcept
{
    if (value == 0)
    {
        return T{0};
    }

    for (std::size_t shift = 1; shift < BitCount<T>; shift <<= 1)
    {
        value = static_cast<T>(value | (value >> shift));
    }

    return static_cast<T>(value - (value >> 1));
}

} // namespace rad::fallback
