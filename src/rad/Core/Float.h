#pragma once

#include <bit>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>

namespace rad
{

template <typename T>
concept FloatingPoint = std::floating_point<T>;

using Float32 = float;
using Float64 = double;

template <FloatingPoint T>
inline constexpr T Pi = T{3.141592653589793238462643383279502884L};

template <FloatingPoint T>
inline constexpr T TwoPi = T{2} * Pi<T>;

template <FloatingPoint T>
inline constexpr T HalfPi = Pi<T> / T{2};

template <FloatingPoint T>
inline constexpr T FloatEpsilon = std::numeric_limits<T>::epsilon();

[[nodiscard]] constexpr float Float32FromBits(std::uint32_t bits) noexcept
{
    return std::bit_cast<float>(bits);
}

[[nodiscard]] constexpr std::uint32_t Float32ToBits(float value) noexcept
{
    return std::bit_cast<std::uint32_t>(value);
}

[[nodiscard]] constexpr double Float64FromBits(std::uint64_t bits) noexcept
{
    return std::bit_cast<double>(bits);
}

[[nodiscard]] constexpr std::uint64_t Float64ToBits(double value) noexcept
{
    return std::bit_cast<std::uint64_t>(value);
}

template <FloatingPoint T>
[[nodiscard]] constexpr T Abs(T value) noexcept
{
    return value < T{0} ? -value : value;
}

template <FloatingPoint T>
[[nodiscard]] constexpr T Min(T lhs, T rhs) noexcept
{
    return lhs < rhs ? lhs : rhs;
}

template <FloatingPoint T>
[[nodiscard]] constexpr T Max(T lhs, T rhs) noexcept
{
    return lhs > rhs ? lhs : rhs;
}

template <FloatingPoint T>
[[nodiscard]] constexpr T Clamp(T value, T minimum, T maximum) noexcept
{
    assert(minimum <= maximum);
    return value < minimum ? minimum : (maximum < value ? maximum : value);
}

template <FloatingPoint T>
[[nodiscard]] constexpr T Saturate(T value) noexcept
{
    return Clamp(value, T{0}, T{1});
}

template <FloatingPoint T>
[[nodiscard]] constexpr bool AlmostZero(T value, T epsilon = FloatEpsilon<T>) noexcept
{
    assert(epsilon >= T{0});
    return Abs(value) <= epsilon;
}

template <FloatingPoint T>
[[nodiscard]] constexpr bool AlmostEqual(T lhs, T rhs, T epsilon = FloatEpsilon<T>) noexcept
{
    assert(epsilon >= T{0});

    if (lhs == rhs)
    {
        return true;
    }

    if ((lhs != lhs) || (rhs != rhs))
    {
        return false;
    }

    const T infinity = std::numeric_limits<T>::infinity();
    if ((Abs(lhs) == infinity) || (Abs(rhs) == infinity))
    {
        return false;
    }

    const T difference = Abs(lhs - rhs);
    const T scale = Max(T{1}, Max(Abs(lhs), Abs(rhs)));
    return difference <= scale * epsilon;
}

template <FloatingPoint T>
[[nodiscard]] constexpr T Lerp(T from, T to, T t) noexcept
{
    return from + (to - from) * t;
}

template <FloatingPoint T>
[[nodiscard]] constexpr T InverseLerp(T from, T to, T value) noexcept
{
    assert(from != to);
    return (value - from) / (to - from);
}

template <FloatingPoint T>
[[nodiscard]] constexpr T Remap(T inputFrom, T inputTo, T outputFrom, T outputTo, T value) noexcept
{
    return Lerp(outputFrom, outputTo, InverseLerp(inputFrom, inputTo, value));
}

template <FloatingPoint T>
[[nodiscard]] constexpr T Normalize(T value, T minimum, T maximum) noexcept
{
    assert(minimum < maximum);
    return Saturate(InverseLerp(minimum, maximum, value));
}

template <FloatingPoint T>
[[nodiscard]] constexpr T SmoothStep(T edge0, T edge1, T value) noexcept
{
    assert(edge0 < edge1);
    const T t = Saturate(InverseLerp(edge0, edge1, value));
    return t * t * (T{3} - T{2} * t);
}

template <FloatingPoint T>
[[nodiscard]] constexpr T DegreesToRadians(T degrees) noexcept
{
    return degrees * Pi<T> / T{180};
}

template <FloatingPoint T>
[[nodiscard]] constexpr T RadiansToDegrees(T radians) noexcept
{
    return radians * T{180} / Pi<T>;
}

template <std::unsigned_integral T, FloatingPoint F>
[[nodiscard]] constexpr T QuantizeUnorm(F value, F minimum, F maximum) noexcept
{
    assert(value == value);
    assert(minimum == minimum);
    assert(maximum == maximum);
    const F normalized = Normalize(value, minimum, maximum);
    if (normalized <= F{0})
    {
        return T{0};
    }

    if (normalized >= F{1})
    {
        return std::numeric_limits<T>::max();
    }

    const long double scaled = static_cast<long double>(normalized) *
                               static_cast<long double>(std::numeric_limits<T>::max());
    return static_cast<T>(scaled + 0.5L);
}

[[nodiscard]] constexpr std::uint8_t QuantizeUnorm8(float value, float minimum,
                                                    float maximum) noexcept
{
    return QuantizeUnorm<std::uint8_t>(value, minimum, maximum);
}

[[nodiscard]] constexpr std::uint16_t QuantizeUnorm16(float value, float minimum,
                                                      float maximum) noexcept
{
    return QuantizeUnorm<std::uint16_t>(value, minimum, maximum);
}

template <std::unsigned_integral T, FloatingPoint F = float>
[[nodiscard]] constexpr F DequantizeUnorm(T value) noexcept
{
    return static_cast<F>(value) / static_cast<F>(std::numeric_limits<T>::max());
}

[[nodiscard]] constexpr float DequantizeUnorm8(std::uint8_t value) noexcept
{
    return DequantizeUnorm(value);
}

[[nodiscard]] constexpr float DequantizeUnorm16(std::uint16_t value) noexcept
{
    return DequantizeUnorm(value);
}

template <FloatingPoint T>
[[nodiscard]] bool IsFinite(T value) noexcept
{
    return std::isfinite(value);
}

template <FloatingPoint T>
[[nodiscard]] bool IsNaN(T value) noexcept
{
    return std::isnan(value);
}

template <FloatingPoint T>
[[nodiscard]] bool IsInfinite(T value) noexcept
{
    return std::isinf(value);
}

} // namespace rad
