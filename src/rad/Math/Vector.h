#pragma once

#include <rad/Core/Math.h>

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>

namespace rad
{

////////////////////////////////////////////////////////////////////////////////
// Eigen built-in aliases
////////////////////////////////////////////////////////////////////////////////

using Eigen::Vector2d;
using Eigen::Vector2f;
using Eigen::Vector2i;
using Eigen::Vector3d;
using Eigen::Vector3f;
using Eigen::Vector3i;
using Eigen::Vector4d;
using Eigen::Vector4f;
using Eigen::Vector4i;

////////////////////////////////////////////////////////////////////////////////
// Angle conversions
////////////////////////////////////////////////////////////////////////////////

template <typename Derived>
    requires FloatingPoint<typename Derived::Scalar> && (Derived::IsVectorAtCompileTime != 0)
[[nodiscard]] typename Derived::PlainObject DegreesToRadians(const Eigen::MatrixBase<Derived>& degrees)
{
    using Scalar = typename Derived::Scalar;
    return degrees.derived() * Pi<Scalar> / Scalar{180};
}

template <typename Derived>
    requires FloatingPoint<typename Derived::Scalar> && (Derived::IsVectorAtCompileTime != 0)
[[nodiscard]] typename Derived::PlainObject RadiansToDegrees(const Eigen::MatrixBase<Derived>& radians)
{
    using Scalar = typename Derived::Scalar;
    return radians.derived() * Scalar{180} / Pi<Scalar>;
}

} // namespace rad
