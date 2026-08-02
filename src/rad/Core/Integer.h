#pragma once

#include <bit>
#include <cassert>
#include <climits>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace rad
{

template <typename T>
concept Integer = std::integral<T> && !std::same_as<std::remove_cv_t<T>, bool>;

template <typename T>
concept UnsignedInteger = Integer<T> && std::unsigned_integral<T>;

using Sint8 = std::int8_t;
using Sint16 = std::int16_t;
using Sint32 = std::int32_t;
using Sint64 = std::int64_t;

using Uint8 = std::uint8_t;
using Uint16 = std::uint16_t;
using Uint32 = std::uint32_t;
using Uint64 = std::uint64_t;

template <typename T>
constexpr std::size_t BitCount = sizeof(T) * CHAR_BIT;

template <typename T>
    requires UnsignedInteger<T> && (BitCount<T> == 32)
[[nodiscard]] constexpr std::uint16_t Upper16Bits(T value) noexcept
{
    return static_cast<std::uint16_t>(value >> 16);
}

template <typename T>
    requires UnsignedInteger<T> && (BitCount<T> == 32)
[[nodiscard]] constexpr std::uint16_t Lower16Bits(T value) noexcept
{
    return static_cast<std::uint16_t>(value & T{0xFFFF});
}

template <typename T>
    requires UnsignedInteger<T> && (BitCount<T> == 64)
[[nodiscard]] constexpr std::uint32_t Upper32Bits(T value) noexcept
{
    return static_cast<std::uint32_t>(value >> 32);
}

template <typename T>
    requires UnsignedInteger<T> && (BitCount<T> == 64)
[[nodiscard]] constexpr std::uint32_t Lower32Bits(T value) noexcept
{
    return static_cast<std::uint32_t>(value & T{0xFFFFFFFF});
}

template <Integer T>
[[nodiscard]] constexpr bool IsPowerOfTwo(T value) noexcept
{
    if constexpr (std::signed_integral<T>)
    {
        if (value <= 0)
        {
            return false;
        }
    }
    else if (value == 0)
    {
        return false;
    }

    using UnsignedT = std::make_unsigned_t<T>;
    const auto unsignedValue = static_cast<UnsignedT>(value);
    return (unsignedValue & (unsignedValue - 1)) == 0;
}

template <UnsignedInteger T>
[[nodiscard]] constexpr bool HasBits(T mask, T bits) noexcept
{
    return (mask & bits) == bits;
}

template <UnsignedInteger T>
[[nodiscard]] constexpr bool HasAnyBits(T mask, T bits) noexcept
{
    return (mask & bits) != 0;
}

template <UnsignedInteger T>
[[nodiscard]] constexpr bool HasNoBits(T mask, T bits) noexcept
{
    return (mask & bits) == 0;
}

// Interprets the low `bits` bits of value as signed and extends the sign.
template <UnsignedInteger T>
[[nodiscard]] constexpr std::make_signed_t<T> SignExtend(T value, std::size_t bits) noexcept
{
    assert(bits > 0);
    assert(bits <= BitCount<T>);
    using SignedT = std::make_signed_t<T>;

    const T valueMask =
        (bits == BitCount<T>) ? std::numeric_limits<T>::max() : static_cast<T>((T{1} << bits) - 1);
    value = static_cast<T>(value & valueMask);

    const T signBit = T{1} << (bits - 1);
    if ((value & signBit) == 0)
    {
        return static_cast<SignedT>(value);
    }

    const T magnitude = static_cast<T>(((~value) & valueMask) + 1);
    if ((bits == BitCount<T>) && (magnitude == signBit))
    {
        return std::numeric_limits<SignedT>::min();
    }

    return static_cast<SignedT>(-static_cast<SignedT>(magnitude));
}

// Extracts `count` bits starting at `start`; bit 0 is the least significant bit.
template <UnsignedInteger T>
[[nodiscard]] constexpr T ExtractBits(T value, std::size_t start, std::size_t count) noexcept
{
    assert(count <= BitCount<T>);
    assert(start <= BitCount<T> - count);
    if (count == 0)
    {
        return T{0};
    }

    if (count == BitCount<T>)
    {
        return value;
    }

    const T mask = static_cast<T>((T{1} << count) - 1);
    return static_cast<T>((value >> start) & mask);
}

// Replaces `count` bits in destination starting at `start` with low bits from source.
template <UnsignedInteger T>
[[nodiscard]] constexpr T ReplaceBits(T destination, T source, std::size_t start,
                                      std::size_t count) noexcept
{
    assert(count <= BitCount<T>);
    assert(start <= BitCount<T> - count);
    if (count == 0)
    {
        return destination;
    }

    if (count == BitCount<T>)
    {
        return source;
    }

    const T mask = static_cast<T>(((T{1} << count) - 1) << start);
    return static_cast<T>((destination & ~mask) | ((source << start) & mask));
}

// Returns the zero-based index of the most significant set bit. Requires value != 0.
template <UnsignedInteger T>
[[nodiscard]] constexpr int BitScanReverse(T value) noexcept
{
    assert(value != 0);
    return static_cast<int>(std::bit_width(value) - 1);
}

template <UnsignedInteger T>
[[nodiscard]] constexpr int CountSetBits(T value) noexcept
{
    return std::popcount(value);
}

template <UnsignedInteger T>
[[nodiscard]] constexpr T ReverseBits(T value) noexcept
{
    T result = 0;
    for (std::size_t bit = 0; bit < BitCount<T>; ++bit)
    {
        result = static_cast<T>((result << 1) | (value & T{1}));
        value = static_cast<T>(value >> 1);
    }

    return result;
}

template <Integer T>
[[nodiscard]] constexpr bool AddWouldOverflow(T lhs, T rhs) noexcept
{
    if constexpr (std::unsigned_integral<T>)
    {
        return lhs > std::numeric_limits<T>::max() - rhs;
    }
    else
    {
        return (rhs > 0 && lhs > std::numeric_limits<T>::max() - rhs) ||
               (rhs < 0 && lhs < std::numeric_limits<T>::min() - rhs);
    }
}

template <Integer T>
[[nodiscard]] constexpr std::optional<T> CheckedAdd(T lhs, T rhs) noexcept
{
    if (AddWouldOverflow(lhs, rhs))
    {
        return std::nullopt;
    }

    return static_cast<T>(lhs + rhs);
}

template <Integer T>
[[nodiscard]] constexpr bool SubtractWouldOverflow(T lhs, T rhs) noexcept
{
    if constexpr (std::unsigned_integral<T>)
    {
        return lhs < rhs;
    }
    else
    {
        return (rhs > 0 && lhs < std::numeric_limits<T>::min() + rhs) ||
               (rhs < 0 && lhs > std::numeric_limits<T>::max() + rhs);
    }
}

template <Integer T>
[[nodiscard]] constexpr std::optional<T> CheckedSubtract(T lhs, T rhs) noexcept
{
    if (SubtractWouldOverflow(lhs, rhs))
    {
        return std::nullopt;
    }

    return static_cast<T>(lhs - rhs);
}

// Requires denominator to be nonzero.
template <UnsignedInteger T>
[[nodiscard]] constexpr T DivideRoundUp(T numerator, T denominator) noexcept
{
    assert(denominator > 0);
    return static_cast<T>(numerator / denominator + (numerator % denominator == 0 ? 0 : 1));
}

// Requires alignment to be a nonzero power of two.
template <UnsignedInteger T>
[[nodiscard]] constexpr T AlignUpToPowerOfTwo(T value, T alignment) noexcept
{
    assert(IsPowerOfTwo(alignment));
    const T mask = static_cast<T>(alignment - 1);
    assert(value <= std::numeric_limits<T>::max() - mask);
    return static_cast<T>((value + mask) & ~mask);
}

// Requires alignment to be a nonzero power of two.
template <UnsignedInteger T>
[[nodiscard]] constexpr T AlignDownToPowerOfTwo(T value, T alignment) noexcept
{
    assert(IsPowerOfTwo(alignment));
    return static_cast<T>(value & ~(alignment - 1));
}

// Requires multiple to be nonzero.
template <UnsignedInteger T>
[[nodiscard]] constexpr T RoundDownToMultiple(T value, T multiple) noexcept
{
    assert(multiple > 0);
    return static_cast<T>(value - value % multiple);
}

// Requires multiple to be nonzero and the rounded result to fit in T.
template <UnsignedInteger T>
[[nodiscard]] constexpr T RoundUpToMultiple(T value, T multiple) noexcept
{
    assert(multiple > 0);
    const T remainder = value % multiple;
    if (remainder == 0)
    {
        return value;
    }

    const T increment = static_cast<T>(multiple - remainder);
    assert(value <= std::numeric_limits<T>::max() - increment);
    return static_cast<T>(value + increment);
}

// Returns the next larger power of two. For power-of-two inputs, this returns the following power.
template <UnsignedInteger T>
[[nodiscard]] constexpr T RoundUpToNextPowerOfTwo(T value) noexcept
{
    assert(value < (T{1} << (BitCount<T> - 1)));
    return static_cast<T>(T{1} << std::bit_width(value));
}

// Returns the smallest power of two that is not smaller than value.
template <UnsignedInteger T>
[[nodiscard]] constexpr T RoundUpToPowerOfTwo(T value) noexcept
{
    assert(value <= (T{1} << (BitCount<T> - 1)));
    return std::bit_ceil(value);
}

// Returns the largest power of two that is not greater than value, or zero for zero.
template <UnsignedInteger T>
[[nodiscard]] constexpr T RoundDownToPowerOfTwo(T value) noexcept
{
    return std::bit_floor(value);
}

namespace literals
{

[[nodiscard]] consteval Uint64 operator""_KiB(unsigned long long value) noexcept
{
    return static_cast<Uint64>(value) * 1024ULL;
}

[[nodiscard]] consteval Uint64 operator""_MiB(unsigned long long value) noexcept
{
    return static_cast<Uint64>(value) * 1024ULL * 1024ULL;
}

[[nodiscard]] consteval Uint64 operator""_GiB(unsigned long long value) noexcept
{
    return static_cast<Uint64>(value) * 1024ULL * 1024ULL * 1024ULL;
}

[[nodiscard]] consteval Uint64 operator""_TiB(unsigned long long value) noexcept
{
    return static_cast<Uint64>(value) * 1024ULL * 1024ULL * 1024ULL * 1024ULL;
}

[[nodiscard]] consteval Uint64 operator""_PiB(unsigned long long value) noexcept
{
    return static_cast<Uint64>(value) * 1024ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL;
}

[[nodiscard]] consteval Uint64 operator""_EiB(unsigned long long value) noexcept
{
    return static_cast<Uint64>(value) * 1024ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL;
}

} // namespace literals

} // namespace rad
