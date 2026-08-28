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
[[nodiscard]] constexpr bool AlmostZero(T value, T epsilon = FloatEpsilon<T>) noexcept
{
    assert(epsilon >= T{0});
    const T magnitude = value < T{0} ? -value : value;
    return magnitude <= epsilon;
}

template <FloatingPoint T>
[[nodiscard]] constexpr bool AlmostEqual(T lhs, T rhs, T epsilon = FloatEpsilon<T>) noexcept
{
    assert(epsilon >= T{0});
    const auto abs = [](T value) constexpr { return value < T{0} ? -value : value; };

    if (lhs == rhs)
    {
        return true;
    }

    if ((lhs != lhs) || (rhs != rhs))
    {
        return false;
    }

    const T infinity = std::numeric_limits<T>::infinity();
    if ((abs(lhs) == infinity) || (abs(rhs) == infinity))
    {
        return false;
    }

    const T difference = abs(lhs - rhs);
    const T maxMagnitude = abs(lhs) > abs(rhs) ? abs(lhs) : abs(rhs);
    const T scale = T{1} > maxMagnitude ? T{1} : maxMagnitude;
    return difference <= scale * epsilon;
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
