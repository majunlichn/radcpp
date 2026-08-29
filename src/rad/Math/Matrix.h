#pragma once

#include <rad/Math/Vector.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rad
{

////////////////////////////////////////////////////////////////////////////////
// Matrix aliases
////////////////////////////////////////////////////////////////////////////////

template <typename T, int M, int N>
    requires std::is_trivially_copyable_v<T> && (M > 0) && (N > 0) &&
             (sizeof(Eigen::Matrix<T, M, N>) == static_cast<std::size_t>(M) * N * sizeof(T))
using MatrixMxN = Eigen::Matrix<T, M, N>;

template <typename T>
    requires std::is_trivially_copyable_v<T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using Matrix2 = MatrixMxN<T, 2, 2>;

template <typename T>
using Matrix3 = MatrixMxN<T, 3, 3>;

template <typename T>
using Matrix4 = MatrixMxN<T, 4, 4>;

template <typename T>
using Matrix2x3 = MatrixMxN<T, 2, 3>;

template <typename T>
using Matrix3x2 = MatrixMxN<T, 3, 2>;

template <typename T>
using Matrix2x4 = MatrixMxN<T, 2, 4>;

template <typename T>
using Matrix4x2 = MatrixMxN<T, 4, 2>;

template <typename T>
using Matrix3x4 = MatrixMxN<T, 3, 4>;

template <typename T>
using Matrix4x3 = MatrixMxN<T, 4, 3>;

using Matrix2f = Matrix2<float>;
using Matrix2d = Matrix2<double>;
using Matrix2i = Matrix2<std::int32_t>;

using Matrix3f = Matrix3<float>;
using Matrix3d = Matrix3<double>;
using Matrix3i = Matrix3<std::int32_t>;

using Matrix4f = Matrix4<float>;
using Matrix4d = Matrix4<double>;
using Matrix4i = Matrix4<std::int32_t>;

using Matrix2x3f = Matrix2x3<float>;
using Matrix2x3d = Matrix2x3<double>;
using Matrix2x3i = Matrix2x3<std::int32_t>;

using Matrix3x2f = Matrix3x2<float>;
using Matrix3x2d = Matrix3x2<double>;
using Matrix3x2i = Matrix3x2<std::int32_t>;

using Matrix2x4f = Matrix2x4<float>;
using Matrix2x4d = Matrix2x4<double>;
using Matrix2x4i = Matrix2x4<std::int32_t>;

using Matrix4x2f = Matrix4x2<float>;
using Matrix4x2d = Matrix4x2<double>;
using Matrix4x2i = Matrix4x2<std::int32_t>;

using Matrix3x4f = Matrix3x4<float>;
using Matrix3x4d = Matrix3x4<double>;
using Matrix3x4i = Matrix3x4<std::int32_t>;

using Matrix4x3f = Matrix4x3<float>;
using Matrix4x3d = Matrix4x3<double>;
using Matrix4x3i = Matrix4x3<std::int32_t>;

////////////////////////////////////////////////////////////////////////////////
// Matrix functions
////////////////////////////////////////////////////////////////////////////////

template <typename Derived>
concept MatrixExpression = (Derived::IsVectorAtCompileTime == 0);

template <typename Derived>
concept ArithmeticMatrix =
    MatrixExpression<Derived> && std::is_arithmetic_v<typename Derived::Scalar> &&
    (!std::same_as<typename Derived::Scalar, bool>);

template <typename Derived>
concept FloatingPointMatrix = MatrixExpression<Derived> && FloatingPoint<typename Derived::Scalar>;

template <typename Derived>
concept SquareMatrix =
    MatrixExpression<Derived> && (Derived::RowsAtCompileTime == Derived::ColsAtCompileTime);

template <typename Derived>
using TransposedMatrix = Eigen::Matrix<typename Derived::Scalar, Derived::ColsAtCompileTime,
                                       Derived::RowsAtCompileTime>;

template <ArithmeticMatrix LhsDerived, ArithmeticMatrix RhsDerived>
    requires std::same_as<typename LhsDerived::Scalar, typename RhsDerived::Scalar> &&
             (LhsDerived::RowsAtCompileTime == Eigen::Dynamic ||
              RhsDerived::RowsAtCompileTime == Eigen::Dynamic ||
              LhsDerived::RowsAtCompileTime == RhsDerived::RowsAtCompileTime) &&
             (LhsDerived::ColsAtCompileTime == Eigen::Dynamic ||
              RhsDerived::ColsAtCompileTime == Eigen::Dynamic ||
              LhsDerived::ColsAtCompileTime == RhsDerived::ColsAtCompileTime)
[[nodiscard]] typename LhsDerived::PlainObject
HadamardProduct(const Eigen::MatrixBase<LhsDerived>& lhs,
                const Eigen::MatrixBase<RhsDerived>& rhs)
{
    assert(lhs.rows() == rhs.rows());
    assert(lhs.cols() == rhs.cols());
    return lhs.cwiseProduct(rhs);
}

template <ArithmeticVector LhsDerived, ArithmeticVector RhsDerived>
    requires std::same_as<typename LhsDerived::Scalar, typename RhsDerived::Scalar> &&
             (LhsDerived::ColsAtCompileTime == 1) && (RhsDerived::ColsAtCompileTime == 1)
[[nodiscard]] Eigen::Matrix<typename LhsDerived::Scalar, LhsDerived::SizeAtCompileTime,
                            RhsDerived::SizeAtCompileTime>
OuterProduct(const Eigen::MatrixBase<LhsDerived>& lhs, const Eigen::MatrixBase<RhsDerived>& rhs)
{
    return lhs.derived() * rhs.derived().transpose();
}

template <ArithmeticMatrix Derived>
[[nodiscard]] TransposedMatrix<Derived> Transpose(const Eigen::MatrixBase<Derived>& matrix)
{
    return matrix.derived().transpose();
}

template <ArithmeticMatrix Derived>
    requires SquareMatrix<Derived>
[[nodiscard]] typename Derived::Scalar Determinant(const Eigen::MatrixBase<Derived>& matrix)
{
    assert(matrix.rows() == matrix.cols());
    return matrix.determinant();
}

template <FloatingPointMatrix Derived>
    requires SquareMatrix<Derived>
[[nodiscard]] typename Derived::PlainObject Inverse(const Eigen::MatrixBase<Derived>& matrix)
{
    assert(matrix.rows() == matrix.cols());
    return matrix.inverse();
}

} // namespace rad
