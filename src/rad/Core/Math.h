#pragma once

#include <rad/Core/Float.h>

namespace rad
{

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

} // namespace rad
