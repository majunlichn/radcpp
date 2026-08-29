#pragma once

#include <rad/Core/Float16.h>
#include <rad/Core/Math.h>

#include <Eigen/Core>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rad
{

////////////////////////////////////////////////////////////////////////////////
// Vector aliases
////////////////////////////////////////////////////////////////////////////////

template <typename T, int Size>
    requires std::is_trivially_copyable_v<T> && (Size > 0) &&
             (sizeof(Eigen::Matrix<T, Size, 1>) == static_cast<std::size_t>(Size) * sizeof(T))
using VectorN = Eigen::Matrix<T, Size, 1>;

template <typename T>
    requires std::is_trivially_copyable_v<T>
using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using Vector2 = VectorN<T, 2>;

template <typename T>
using Vector3 = VectorN<T, 3>;

template <typename T>
using Vector4 = VectorN<T, 4>;

////////////////////////////////////////////////////////////////////////////////
// Angle conversions
////////////////////////////////////////////////////////////////////////////////

template <typename Derived>
concept VectorExpression = (Derived::IsVectorAtCompileTime != 0);

template <typename Derived>
concept ArithmeticVector =
    VectorExpression<Derived> && std::is_arithmetic_v<typename Derived::Scalar> &&
    (!std::same_as<typename Derived::Scalar, bool>);

template <typename Derived>
concept FloatingPointVector = VectorExpression<Derived> && FloatingPoint<typename Derived::Scalar>;

template <typename Derived>
concept Sint32Vector =
    VectorExpression<Derived> && std::same_as<typename Derived::Scalar, std::int32_t>;

template <typename Derived>
concept Uint32Vector =
    VectorExpression<Derived> && std::same_as<typename Derived::Scalar, std::uint32_t>;

template <typename Derived>
concept BoolVector = VectorExpression<Derived> && std::same_as<typename Derived::Scalar, bool>;

template <typename Derived>
concept EqualityComparableVector =
    VectorExpression<Derived> && std::equality_comparable<typename Derived::Scalar>;

template <typename Derived>
using BoolMask = Eigen::Matrix<bool, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;

template <typename Derived, int Size>
concept VectorOfSize =
    VectorExpression<Derived> && (Derived::SizeAtCompileTime == Size);

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Radians(const Eigen::MatrixBase<Derived>& degrees)
{
    using Scalar = typename Derived::Scalar;
    return degrees.derived() * Pi<Scalar> / Scalar{180};
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Degrees(const Eigen::MatrixBase<Derived>& radians)
{
    using Scalar = typename Derived::Scalar;
    return radians.derived() * Scalar{180} / Pi<Scalar>;
}

////////////////////////////////////////////////////////////////////////////////
// Trigonometric functions
////////////////////////////////////////////////////////////////////////////////

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Sin(const Eigen::MatrixBase<Derived>& angles)
{
    return angles.array().sin().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Cos(const Eigen::MatrixBase<Derived>& angles)
{
    return angles.array().cos().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Tan(const Eigen::MatrixBase<Derived>& angles)
{
    return angles.array().tan().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Asin(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().asin().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Acos(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().acos().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Atan(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().atan().matrix();
}

template <FloatingPointVector YDerived, FloatingPointVector XDerived>
    requires std::same_as<typename YDerived::Scalar, typename XDerived::Scalar>
[[nodiscard]] typename YDerived::PlainObject Atan2(const Eigen::MatrixBase<YDerived>& y,
                                                   const Eigen::MatrixBase<XDerived>& x)
{
    return y.array().atan2(x.array()).matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Sinh(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().sinh().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Cosh(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().cosh().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Tanh(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().tanh().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Asinh(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().asinh().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Acosh(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().acosh().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Atanh(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().atanh().matrix();
}

////////////////////////////////////////////////////////////////////////////////
// Exponential functions
////////////////////////////////////////////////////////////////////////////////

template <FloatingPointVector BaseDerived, FloatingPointVector ExponentDerived>
    requires std::same_as<typename BaseDerived::Scalar, typename ExponentDerived::Scalar>
[[nodiscard]] typename BaseDerived::PlainObject Pow(const Eigen::MatrixBase<BaseDerived>& bases,
                                                    const Eigen::MatrixBase<ExponentDerived>& exponents)
{
    return bases.array().pow(exponents.array()).matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Exp(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().exp().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Log(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().log().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Exp2(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().exp2().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Log2(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().log2().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Log10(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().log10().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Sqrt(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().sqrt().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Rsqrt(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().rsqrt().matrix();
}

////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////

template <ArithmeticVector Derived>
[[nodiscard]] typename Derived::PlainObject Abs(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().abs().matrix();
}

template <ArithmeticVector Derived>
[[nodiscard]] typename Derived::PlainObject Sign(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().sign().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Floor(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().floor().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Trunc(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().trunc().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Round(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().round().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject RoundEven(const Eigen::MatrixBase<Derived>& values)
{
    using Scalar = typename Derived::Scalar;
    return values.unaryExpr([](Scalar value) { return rad::RoundEven(value); });
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Ceil(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().ceil().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Fract(const Eigen::MatrixBase<Derived>& values)
{
    return (values.array() - values.array().floor()).matrix();
}

template <FloatingPointVector XDerived, FloatingPointVector YDerived>
    requires std::same_as<typename XDerived::Scalar, typename YDerived::Scalar>
[[nodiscard]] typename XDerived::PlainObject Mod(const Eigen::MatrixBase<XDerived>& x,
                                                 const Eigen::MatrixBase<YDerived>& y)
{
    return (x.array() - y.array() * (x.array() / y.array()).floor()).matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Mod(const Eigen::MatrixBase<Derived>& x,
                                                typename Derived::Scalar y)
{
    return (x.array() - y * (x.array() / y).floor()).matrix();
}

template <FloatingPointVector Derived, FloatingPointVector IntegralDerived>
    requires std::same_as<typename Derived::Scalar, typename IntegralDerived::Scalar>
[[nodiscard]] typename Derived::PlainObject Modf(const Eigen::MatrixBase<Derived>& values,
                                                 Eigen::MatrixBase<IntegralDerived>& integralParts)
{
    assert(values.size() == integralParts.size());
    typename Derived::PlainObject fractions(values.rows(), values.cols());
    for (Eigen::Index i = 0; i < values.size(); ++i)
    {
        typename Derived::Scalar integralPart;
        fractions(i) = std::modf(values(i), &integralPart);
        integralParts.derived()(i) = integralPart;
    }
    return fractions;
}

template <ArithmeticVector XDerived, ArithmeticVector YDerived>
    requires std::same_as<typename XDerived::Scalar, typename YDerived::Scalar>
[[nodiscard]] typename XDerived::PlainObject Min(const Eigen::MatrixBase<XDerived>& x,
                                                 const Eigen::MatrixBase<YDerived>& y)
{
    return x.cwiseMin(y);
}

template <ArithmeticVector Derived>
[[nodiscard]] typename Derived::PlainObject Min(const Eigen::MatrixBase<Derived>& x,
                                                typename Derived::Scalar y)
{
    return x.cwiseMin(y);
}

template <ArithmeticVector XDerived, ArithmeticVector YDerived>
    requires std::same_as<typename XDerived::Scalar, typename YDerived::Scalar>
[[nodiscard]] typename XDerived::PlainObject Max(const Eigen::MatrixBase<XDerived>& x,
                                                 const Eigen::MatrixBase<YDerived>& y)
{
    return x.cwiseMax(y);
}

template <ArithmeticVector Derived>
[[nodiscard]] typename Derived::PlainObject Max(const Eigen::MatrixBase<Derived>& x,
                                                typename Derived::Scalar y)
{
    return x.cwiseMax(y);
}

template <ArithmeticVector XDerived, ArithmeticVector MinDerived, ArithmeticVector MaxDerived>
    requires std::same_as<typename XDerived::Scalar, typename MinDerived::Scalar> &&
             std::same_as<typename XDerived::Scalar, typename MaxDerived::Scalar>
[[nodiscard]] typename XDerived::PlainObject Clamp(const Eigen::MatrixBase<XDerived>& x,
                                                   const Eigen::MatrixBase<MinDerived>& minimum,
                                                   const Eigen::MatrixBase<MaxDerived>& maximum)
{
    return x.cwiseMax(minimum).cwiseMin(maximum);
}

template <ArithmeticVector Derived>
[[nodiscard]] typename Derived::PlainObject Clamp(const Eigen::MatrixBase<Derived>& x,
                                                  typename Derived::Scalar minimum,
                                                  typename Derived::Scalar maximum)
{
    return x.cwiseMax(minimum).cwiseMin(maximum);
}

template <FloatingPointVector XDerived, FloatingPointVector YDerived, FloatingPointVector ADerived>
    requires std::same_as<typename XDerived::Scalar, typename YDerived::Scalar> &&
             std::same_as<typename XDerived::Scalar, typename ADerived::Scalar>
[[nodiscard]] typename XDerived::PlainObject Mix(const Eigen::MatrixBase<XDerived>& x,
                                                 const Eigen::MatrixBase<YDerived>& y,
                                                 const Eigen::MatrixBase<ADerived>& a)
{
    return (x.array() + a.array() * (y.array() - x.array())).matrix();
}

template <FloatingPointVector XDerived, FloatingPointVector YDerived>
    requires std::same_as<typename XDerived::Scalar, typename YDerived::Scalar>
[[nodiscard]] typename XDerived::PlainObject Mix(const Eigen::MatrixBase<XDerived>& x,
                                                 const Eigen::MatrixBase<YDerived>& y,
                                                 typename XDerived::Scalar a)
{
    return x + a * (y - x);
}

template <FloatingPointVector EdgeDerived, FloatingPointVector XDerived>
    requires std::same_as<typename EdgeDerived::Scalar, typename XDerived::Scalar>
[[nodiscard]] typename XDerived::PlainObject Step(const Eigen::MatrixBase<EdgeDerived>& edge,
                                                  const Eigen::MatrixBase<XDerived>& x)
{
    using Scalar = typename XDerived::Scalar;
    assert(edge.size() == x.size());
    typename XDerived::PlainObject result(x.rows(), x.cols());
    for (Eigen::Index i = 0; i < x.size(); ++i)
    {
        result(i) = x(i) < edge(i) ? Scalar{0} : Scalar{1};
    }
    return result;
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Step(typename Derived::Scalar edge,
                                                 const Eigen::MatrixBase<Derived>& x)
{
    using Scalar = typename Derived::Scalar;
    return x.unaryExpr(
        [edge](Scalar value) { return value < edge ? Scalar{0} : Scalar{1}; });
}

template <FloatingPointVector Edge0Derived, FloatingPointVector Edge1Derived,
          FloatingPointVector XDerived>
    requires std::same_as<typename Edge0Derived::Scalar, typename Edge1Derived::Scalar> &&
             std::same_as<typename Edge0Derived::Scalar, typename XDerived::Scalar>
[[nodiscard]] typename XDerived::PlainObject SmoothStep(
    const Eigen::MatrixBase<Edge0Derived>& edge0,
    const Eigen::MatrixBase<Edge1Derived>& edge1,
    const Eigen::MatrixBase<XDerived>& x)
{
    using Scalar = typename XDerived::Scalar;
    const auto t = ((x.array() - edge0.array()) / (edge1.array() - edge0.array()))
                       .max(Scalar{0})
                       .min(Scalar{1});
    return (t * t * (Scalar{3} - Scalar{2} * t)).matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject SmoothStep(typename Derived::Scalar edge0,
                                                        typename Derived::Scalar edge1,
                                                        const Eigen::MatrixBase<Derived>& x)
{
    using Scalar = typename Derived::Scalar;
    const auto t = ((x.array() - edge0) / (edge1 - edge0)).max(Scalar{0}).min(Scalar{1});
    return (t * t * (Scalar{3} - Scalar{2} * t)).matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] BoolMask<Derived> IsNaN(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().isNaN().matrix();
}

template <FloatingPointVector Derived>
[[nodiscard]] BoolMask<Derived> IsInfinite(const Eigen::MatrixBase<Derived>& values)
{
    return values.array().isInf().matrix();
}

template <FloatingPointVector ADerived, FloatingPointVector BDerived,
          FloatingPointVector CDerived>
    requires std::same_as<typename ADerived::Scalar, typename BDerived::Scalar> &&
             std::same_as<typename ADerived::Scalar, typename CDerived::Scalar>
[[nodiscard]] typename ADerived::PlainObject Fma(const Eigen::MatrixBase<ADerived>& a,
                                                 const Eigen::MatrixBase<BDerived>& b,
                                                 const Eigen::MatrixBase<CDerived>& c)
{
    assert((a.size() == b.size()) && (a.size() == c.size()));
    typename ADerived::PlainObject result(a.rows(), a.cols());
    for (Eigen::Index i = 0; i < a.size(); ++i)
    {
        result(i) = std::fma(a(i), b(i), c(i));
    }
    return result;
}

template <FloatingPointVector Derived, Sint32Vector ExponentDerived>
[[nodiscard]] typename Derived::PlainObject Frexp(const Eigen::MatrixBase<Derived>& values,
                                                  Eigen::MatrixBase<ExponentDerived>& exponents)
{
    assert(values.size() == exponents.size());
    typename Derived::PlainObject significands(values.rows(), values.cols());
    for (Eigen::Index i = 0; i < values.size(); ++i)
    {
        int exponent;
        significands(i) = std::frexp(values(i), &exponent);
        exponents.derived()(i) = exponent;
    }
    return significands;
}

template <FloatingPointVector Derived, Sint32Vector ExponentDerived>
[[nodiscard]] typename Derived::PlainObject Ldexp(const Eigen::MatrixBase<Derived>& values,
                                                  const Eigen::MatrixBase<ExponentDerived>& exponents)
{
    assert(values.size() == exponents.size());
    typename Derived::PlainObject result(values.rows(), values.cols());
    for (Eigen::Index i = 0; i < values.size(); ++i)
    {
        result(i) = std::ldexp(values(i), exponents(i));
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Geometry functions
////////////////////////////////////////////////////////////////////////////////

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::Scalar Length(const Eigen::MatrixBase<Derived>& value)
{
    return value.norm();
}

template <FloatingPointVector P0Derived, FloatingPointVector P1Derived>
    requires std::same_as<typename P0Derived::Scalar, typename P1Derived::Scalar>
[[nodiscard]] typename P0Derived::Scalar Distance(const Eigen::MatrixBase<P0Derived>& p0,
                                                  const Eigen::MatrixBase<P1Derived>& p1)
{
    assert(p0.size() == p1.size());
    return (p0 - p1).norm();
}

template <FloatingPointVector XDerived, FloatingPointVector YDerived>
    requires std::same_as<typename XDerived::Scalar, typename YDerived::Scalar>
[[nodiscard]] typename XDerived::Scalar Dot(const Eigen::MatrixBase<XDerived>& x,
                                            const Eigen::MatrixBase<YDerived>& y)
{
    assert(x.size() == y.size());
    return x.dot(y);
}

template <FloatingPointVector XDerived, FloatingPointVector YDerived>
    requires std::same_as<typename XDerived::Scalar, typename YDerived::Scalar> &&
             (XDerived::SizeAtCompileTime == 3) && (YDerived::SizeAtCompileTime == 3)
[[nodiscard]] typename XDerived::PlainObject Cross(const Eigen::MatrixBase<XDerived>& x,
                                                   const Eigen::MatrixBase<YDerived>& y)
{
    typename XDerived::PlainObject result;
    result(0) = x(1) * y(2) - x(2) * y(1);
    result(1) = x(2) * y(0) - x(0) * y(2);
    result(2) = x(0) * y(1) - x(1) * y(0);
    return result;
}

template <FloatingPointVector Derived>
[[nodiscard]] typename Derived::PlainObject Normalize(const Eigen::MatrixBase<Derived>& value)
{
    return value.normalized();
}

template <FloatingPointVector NDerived, FloatingPointVector IDerived,
          FloatingPointVector NrefDerived>
    requires std::same_as<typename NDerived::Scalar, typename IDerived::Scalar> &&
             std::same_as<typename NDerived::Scalar, typename NrefDerived::Scalar>
[[nodiscard]] typename NDerived::PlainObject FaceForward(const Eigen::MatrixBase<NDerived>& normal,
                                                         const Eigen::MatrixBase<IDerived>& incident,
                                                         const Eigen::MatrixBase<NrefDerived>& referenceNormal)
{
    assert((normal.size() == incident.size()) && (normal.size() == referenceNormal.size()));
    return referenceNormal.dot(incident) < typename NDerived::Scalar{0}
               ? typename NDerived::PlainObject{normal}
               : typename NDerived::PlainObject{-normal};
}

template <FloatingPointVector IDerived, FloatingPointVector NDerived>
    requires std::same_as<typename IDerived::Scalar, typename NDerived::Scalar>
[[nodiscard]] typename IDerived::PlainObject Reflect(const Eigen::MatrixBase<IDerived>& incident,
                                                     const Eigen::MatrixBase<NDerived>& normal)
{
    assert(incident.size() == normal.size());
    using Scalar = typename IDerived::Scalar;
    return incident - Scalar{2} * normal.dot(incident) * normal;
}

template <FloatingPointVector IDerived, FloatingPointVector NDerived>
    requires std::same_as<typename IDerived::Scalar, typename NDerived::Scalar>
[[nodiscard]] typename IDerived::PlainObject Refract(const Eigen::MatrixBase<IDerived>& incident,
                                                     const Eigen::MatrixBase<NDerived>& normal,
                                                     typename IDerived::Scalar eta)
{
    assert(incident.size() == normal.size());
    using Scalar = typename IDerived::Scalar;
    const Scalar dotNI = normal.dot(incident);
    const Scalar k = Scalar{1} - eta * eta * (Scalar{1} - dotNI * dotNI);
    if (k < Scalar{0})
    {
        return IDerived::PlainObject::Zero(incident.rows(), incident.cols());
    }
    return eta * incident - (eta * dotNI + std::sqrt(k)) * normal;
}

////////////////////////////////////////////////////////////////////////////////
// Vector relational functions
////////////////////////////////////////////////////////////////////////////////

template <ArithmeticVector XDerived, ArithmeticVector YDerived>
    requires std::same_as<typename XDerived::Scalar, typename YDerived::Scalar>
[[nodiscard]] BoolMask<XDerived> LessThan(const Eigen::MatrixBase<XDerived>& x,
                                             const Eigen::MatrixBase<YDerived>& y)
{
    assert(x.size() == y.size());
    return (x.array() < y.array()).matrix();
}

template <ArithmeticVector XDerived, ArithmeticVector YDerived>
    requires std::same_as<typename XDerived::Scalar, typename YDerived::Scalar>
[[nodiscard]] BoolMask<XDerived> LessThanEqual(const Eigen::MatrixBase<XDerived>& x,
                                                  const Eigen::MatrixBase<YDerived>& y)
{
    assert(x.size() == y.size());
    return (x.array() <= y.array()).matrix();
}

template <ArithmeticVector XDerived, ArithmeticVector YDerived>
    requires std::same_as<typename XDerived::Scalar, typename YDerived::Scalar>
[[nodiscard]] BoolMask<XDerived> GreaterThan(const Eigen::MatrixBase<XDerived>& x,
                                                const Eigen::MatrixBase<YDerived>& y)
{
    assert(x.size() == y.size());
    return (x.array() > y.array()).matrix();
}

template <ArithmeticVector XDerived, ArithmeticVector YDerived>
    requires std::same_as<typename XDerived::Scalar, typename YDerived::Scalar>
[[nodiscard]] BoolMask<XDerived> GreaterThanEqual(const Eigen::MatrixBase<XDerived>& x,
                                                     const Eigen::MatrixBase<YDerived>& y)
{
    assert(x.size() == y.size());
    return (x.array() >= y.array()).matrix();
}

template <EqualityComparableVector XDerived, EqualityComparableVector YDerived>
    requires std::same_as<typename XDerived::Scalar, typename YDerived::Scalar>
[[nodiscard]] BoolMask<XDerived> Equal(const Eigen::MatrixBase<XDerived>& x,
                                           const Eigen::MatrixBase<YDerived>& y)
{
    assert(x.size() == y.size());
    return (x.array() == y.array()).matrix();
}

template <EqualityComparableVector XDerived, EqualityComparableVector YDerived>
    requires std::same_as<typename XDerived::Scalar, typename YDerived::Scalar>
[[nodiscard]] BoolMask<XDerived> NotEqual(const Eigen::MatrixBase<XDerived>& x,
                                              const Eigen::MatrixBase<YDerived>& y)
{
    assert(x.size() == y.size());
    return (x.array() != y.array()).matrix();
}

template <BoolVector Derived>
[[nodiscard]] bool Any(const Eigen::DenseBase<Derived>& values)
{
    return values.any();
}

template <BoolVector Derived>
[[nodiscard]] bool All(const Eigen::DenseBase<Derived>& values)
{
    return values.all();
}

template <BoolVector Derived>
[[nodiscard]] BoolMask<Derived> Not(const Eigen::DenseBase<Derived>& values)
{
    return (!values.derived().array()).matrix();
}

////////////////////////////////////////////////////////////////////////////////
// Floating-point pack and unpack functions
////////////////////////////////////////////////////////////////////////////////

template <typename Derived>
    requires FloatingPointVector<Derived> && std::same_as<typename Derived::Scalar, float> &&
             VectorOfSize<Derived, 2>
[[nodiscard]] std::uint32_t PackUnorm2x16(const Eigen::MatrixBase<Derived>& values)
{
    const auto x = QuantizeUnorm<std::uint16_t>(values(0), 0.0f, 1.0f);
    const auto y = QuantizeUnorm<std::uint16_t>(values(1), 0.0f, 1.0f);
    return static_cast<std::uint32_t>(x) | (static_cast<std::uint32_t>(y) << 16);
}

template <typename Derived>
    requires FloatingPointVector<Derived> && std::same_as<typename Derived::Scalar, float> &&
             VectorOfSize<Derived, 2>
[[nodiscard]] std::uint32_t PackSnorm2x16(const Eigen::MatrixBase<Derived>& values)
{
    const auto x = QuantizeSnorm<std::int16_t>(values(0), -1.0f, 1.0f);
    const auto y = QuantizeSnorm<std::int16_t>(values(1), -1.0f, 1.0f);
    return static_cast<std::uint16_t>(x) |
           (static_cast<std::uint32_t>(static_cast<std::uint16_t>(y)) << 16);
}

template <typename Derived>
    requires FloatingPointVector<Derived> && std::same_as<typename Derived::Scalar, float> &&
             VectorOfSize<Derived, 4>
[[nodiscard]] std::uint32_t PackUnorm4x8(const Eigen::MatrixBase<Derived>& values)
{
    std::uint32_t packed = 0;
    for (Eigen::Index i = 0; i < 4; ++i)
    {
        const auto component = QuantizeUnorm<std::uint8_t>(values(i), 0.0f, 1.0f);
        packed |= static_cast<std::uint32_t>(component) << (i * 8);
    }
    return packed;
}

template <typename Derived>
    requires FloatingPointVector<Derived> && std::same_as<typename Derived::Scalar, float> &&
             VectorOfSize<Derived, 4>
[[nodiscard]] std::uint32_t PackSnorm4x8(const Eigen::MatrixBase<Derived>& values)
{
    std::uint32_t packed = 0;
    for (Eigen::Index i = 0; i < 4; ++i)
    {
        const auto component = QuantizeSnorm<std::int8_t>(values(i), -1.0f, 1.0f);
        packed |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(component)) << (i * 8);
    }
    return packed;
}

[[nodiscard]] inline Vector2<float> UnpackUnorm2x16(std::uint32_t packed) noexcept
{
    return Vector2<float>{
        DequantizeUnorm<std::uint16_t>(static_cast<std::uint16_t>(packed)),
        DequantizeUnorm<std::uint16_t>(static_cast<std::uint16_t>(packed >> 16))};
}

[[nodiscard]] inline Vector2<float> UnpackSnorm2x16(std::uint32_t packed) noexcept
{
    const auto x = std::bit_cast<std::int16_t>(static_cast<std::uint16_t>(packed));
    const auto y = std::bit_cast<std::int16_t>(static_cast<std::uint16_t>(packed >> 16));
    return Vector2<float>{
        DequantizeSnorm<std::int16_t>(x),
        DequantizeSnorm<std::int16_t>(y)};
}

[[nodiscard]] inline Vector4<float> UnpackUnorm4x8(std::uint32_t packed) noexcept
{
    Vector4<float> result;
    for (Eigen::Index i = 0; i < 4; ++i)
    {
        result(i) =
            DequantizeUnorm<std::uint8_t>(static_cast<std::uint8_t>(packed >> (i * 8)));
    }
    return result;
}

[[nodiscard]] inline Vector4<float> UnpackSnorm4x8(std::uint32_t packed) noexcept
{
    Vector4<float> result;
    for (Eigen::Index i = 0; i < 4; ++i)
    {
        const auto bits = static_cast<std::uint8_t>(packed >> (i * 8));
        result(i) = DequantizeSnorm<std::int8_t>(std::bit_cast<std::int8_t>(bits));
    }
    return result;
}

template <typename Derived>
    requires FloatingPointVector<Derived> && std::same_as<typename Derived::Scalar, float> &&
             VectorOfSize<Derived, 2>
[[nodiscard]] std::uint32_t PackHalf2x16(const Eigen::MatrixBase<Derived>& values) noexcept
{
    const auto x = Float16{values(0)}.bits();
    const auto y = Float16{values(1)}.bits();
    return static_cast<std::uint32_t>(x) | (static_cast<std::uint32_t>(y) << 16);
}

[[nodiscard]] inline Vector2<float> UnpackHalf2x16(std::uint32_t packed) noexcept
{
    Float16 x;
    Float16 y;
    x.setBits(static_cast<std::uint16_t>(packed));
    y.setBits(static_cast<std::uint16_t>(packed >> 16));
    return Vector2<float>{static_cast<float>(x), static_cast<float>(y)};
}

template <typename Derived>
    requires Uint32Vector<Derived> && VectorOfSize<Derived, 2>
[[nodiscard]] double PackDouble2x32(const Eigen::MatrixBase<Derived>& values) noexcept
{
    const std::uint64_t bits =
        static_cast<std::uint64_t>(values(0)) | (static_cast<std::uint64_t>(values(1)) << 32);
    return std::bit_cast<double>(bits);
}

[[nodiscard]] inline Vector2<std::uint32_t> UnpackDouble2x32(double value) noexcept
{
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    return Vector2<std::uint32_t>{
        static_cast<std::uint32_t>(bits),
        static_cast<std::uint32_t>(bits >> 32)};
}

} // namespace rad
