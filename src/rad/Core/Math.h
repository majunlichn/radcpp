#pragma once

#include <rad/Core/Float.h>

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rad
{

template <FloatingPoint T>
inline constexpr T Pi = T{3.141592653589793238462643383279502884L};

template <FloatingPoint T>
inline constexpr T TwoPi = T{2} * Pi<T>;

template <FloatingPoint T>
inline constexpr T HalfPi = Pi<T> / T{2};

template <FloatingPoint T>
[[nodiscard]] T Sin(T value) { return std::sin(value); }

template <FloatingPoint T>
[[nodiscard]] T Cos(T value) { return std::cos(value); }

template <FloatingPoint T>
[[nodiscard]] T Tan(T value) { return std::tan(value); }

template <FloatingPoint T>
[[nodiscard]] T Asin(T value) { return std::asin(value); }

template <FloatingPoint T>
[[nodiscard]] T Acos(T value) { return std::acos(value); }

template <FloatingPoint T>
[[nodiscard]] T Atan(T value) { return std::atan(value); }

template <FloatingPoint T>
[[nodiscard]] T Atan2(T y, T x) { return std::atan2(y, x); }

template <FloatingPoint T>
[[nodiscard]] T Sinh(T value) { return std::sinh(value); }

template <FloatingPoint T>
[[nodiscard]] T Cosh(T value) { return std::cosh(value); }

template <FloatingPoint T>
[[nodiscard]] T Tanh(T value) { return std::tanh(value); }

template <FloatingPoint T>
[[nodiscard]] T Asinh(T value) { return std::asinh(value); }

template <FloatingPoint T>
[[nodiscard]] T Acosh(T value) { return std::acosh(value); }

template <FloatingPoint T>
[[nodiscard]] T Atanh(T value) { return std::atanh(value); }

template <FloatingPoint T>
[[nodiscard]] T Pow(T base, T exponent) { return std::pow(base, exponent); }

template <FloatingPoint T>
[[nodiscard]] T Exp(T value) { return std::exp(value); }

template <FloatingPoint T>
[[nodiscard]] T Log(T value) { return std::log(value); }

template <FloatingPoint T>
[[nodiscard]] T Exp2(T value) { return std::exp2(value); }

template <FloatingPoint T>
[[nodiscard]] T Log2(T value) { return std::log2(value); }

template <FloatingPoint T>
[[nodiscard]] T Log10(T value) { return std::log10(value); }

template <FloatingPoint T>
[[nodiscard]] T Sqrt(T value) { return std::sqrt(value); }

template <FloatingPoint T>
[[nodiscard]] T Rsqrt(T value) { return T{1} / std::sqrt(value); }

template <FloatingPoint T>
[[nodiscard]] constexpr T Abs(T value) noexcept
{
    return value < T{0} ? -value : value;
}

template <FloatingPoint T>
[[nodiscard]] constexpr T Sign(T value) noexcept
{
    return value < T{0} ? T{-1} : (value > T{0} ? T{1} : T{0});
}

template <FloatingPoint T>
[[nodiscard]] T Floor(T value) { return std::floor(value); }

template <FloatingPoint T>
[[nodiscard]] T Trunc(T value) { return std::trunc(value); }

template <FloatingPoint T>
[[nodiscard]] T Round(T value) { return std::round(value); }

template <FloatingPoint T>
[[nodiscard]] T RoundEven(T value)
{
    const T lower = std::floor(value);
    const T fraction = value - lower;
    if (fraction < T{0.5})
    {
        return lower;
    }
    if (fraction > T{0.5})
    {
        return lower + T{1};
    }
    return std::fmod(lower, T{2}) == T{0} ? lower : lower + T{1};
}

template <FloatingPoint T>
[[nodiscard]] T Ceil(T value) { return std::ceil(value); }

template <FloatingPoint T>
[[nodiscard]] T Fract(T value) { return value - std::floor(value); }

template <FloatingPoint T>
[[nodiscard]] T Mod(T x, T y) { return x - y * std::floor(x / y); }

template <FloatingPoint T>
[[nodiscard]] T Modf(T value, T& integralPart)
{
    return std::modf(value, &integralPart);
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
[[nodiscard]] constexpr T Mix(T x, T y, T a) noexcept
{
    return Lerp(x, y, a);
}

template <FloatingPoint T>
[[nodiscard]] constexpr T Step(T edge, T x) noexcept
{
    return x < edge ? T{0} : T{1};
}

template <FloatingPoint T>
[[nodiscard]] constexpr T SmoothStep(T edge0, T edge1, T value) noexcept
{
    assert(edge0 < edge1);
    const T t = Saturate(InverseLerp(edge0, edge1, value));
    return t * t * (T{3} - T{2} * t);
}

template <FloatingPoint T>
[[nodiscard]] T Fma(T a, T b, T c) { return std::fma(a, b, c); }

template <FloatingPoint T>
[[nodiscard]] T Frexp(T value, int& exponent)
{
    return std::frexp(value, &exponent);
}

template <FloatingPoint T>
[[nodiscard]] T Ldexp(T value, int exponent) { return std::ldexp(value, exponent); }

template <FloatingPoint T>
[[nodiscard]] constexpr T Radians(T degrees) noexcept
{
    return degrees * Pi<T> / T{180};
}

template <FloatingPoint T>
[[nodiscard]] constexpr T Degrees(T radians) noexcept
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

    using ComputeType = std::conditional_t<
        (std::numeric_limits<F>::digits >= std::numeric_limits<T>::digits), F, double>;
    const ComputeType scaled = static_cast<ComputeType>(normalized) *
                               static_cast<ComputeType>(std::numeric_limits<T>::max());
    return static_cast<T>(scaled + ComputeType{0.5});
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

template <std::signed_integral T, FloatingPoint F>
[[nodiscard]] constexpr T QuantizeSnorm(F value, F minimum, F maximum) noexcept
{
    assert(value == value);
    assert(minimum == minimum);
    assert(maximum == maximum);
    const F normalized = Normalize(value, minimum, maximum) * F{2} - F{1};
    const T maxMagnitude = std::numeric_limits<T>::max();
    if (normalized <= F{-1})
    {
        return static_cast<T>(-maxMagnitude);
    }
    if (normalized >= F{1})
    {
        return maxMagnitude;
    }

    using ComputeType = std::conditional_t<
        (std::numeric_limits<F>::digits >= std::numeric_limits<T>::digits), F, double>;
    const ComputeType scaled = static_cast<ComputeType>(normalized) *
                               static_cast<ComputeType>(maxMagnitude);
    return static_cast<T>(
        scaled < ComputeType{0} ? scaled - ComputeType{0.5} : scaled + ComputeType{0.5});
}

template <std::signed_integral T, FloatingPoint F = float>
[[nodiscard]] constexpr F DequantizeSnorm(T value) noexcept
{
    const F normalized =
        static_cast<F>(value) / static_cast<F>(std::numeric_limits<T>::max());
    return Max(normalized, F{-1});
}

} // namespace rad
