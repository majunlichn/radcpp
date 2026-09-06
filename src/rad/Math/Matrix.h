#pragma once

#include <rad/Math/Vector.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
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

////////////////////////////////////////////////////////////////////////////////
// Matrix transformations
////////////////////////////////////////////////////////////////////////////////

// Column-vector convention (p' = M * p): each function post-multiplies the
// given matrix, i.e. returns matrix * transform.

// Returns matrix * T where T is a translation by `translation`. Works in any
// dimension D: `matrix` is a (D+1) x (D+1) affine transform of D-space and
// `translation` is a D-vector.
template <FloatingPointMatrix MatrixDerived, FloatingPointVector VectorDerived>
    requires SquareMatrix<MatrixDerived> && (MatrixDerived::RowsAtCompileTime >= 3) &&
             (MatrixDerived::RowsAtCompileTime == VectorDerived::SizeAtCompileTime + 1) &&
             std::same_as<typename MatrixDerived::Scalar, typename VectorDerived::Scalar>
[[nodiscard]] typename MatrixDerived::PlainObject
Translate(const Eigen::MatrixBase<MatrixDerived>& matrix,
          const Eigen::MatrixBase<VectorDerived>& translation)
{
    using Scalar = typename MatrixDerived::Scalar;
    constexpr int dim = MatrixDerived::RowsAtCompileTime - 1;
    return matrix *
           Eigen::Transform<Scalar, dim, Eigen::Isometry>(
               Eigen::Translation<Scalar, dim>(translation.derived().eval()))
               .matrix();
}

// Returns matrix * R where R is the in-plane 2D rotation by `angle` radians,
// embedded as the top-left 2x2 block with identity elsewhere. Works for any
// dimension D >= 2 (in 3D this is a rotation about the Z axis).
template <FloatingPointMatrix MatrixDerived>
    requires SquareMatrix<MatrixDerived> && (MatrixDerived::RowsAtCompileTime >= 3)
[[nodiscard]] typename MatrixDerived::PlainObject
Rotate(const Eigen::MatrixBase<MatrixDerived>& matrix, typename MatrixDerived::Scalar angle)
{
    using Scalar = typename MatrixDerived::Scalar;
    constexpr int dim = MatrixDerived::RowsAtCompileTime - 1;
    typename MatrixDerived::PlainObject rotation =
        Eigen::Matrix<Scalar, dim + 1, dim + 1>::Identity();
    rotation.template topLeftCorner<2, 2>() = Eigen::Rotation2D<Scalar>(angle).toRotationMatrix();
    return matrix * rotation;
}

// Returns matrix * R where R is the right-handed rotation by `angle` radians
// about `axis` in 3D. `axis` need not be normalized (a zero axis leaves the
// matrix unchanged).
template <FloatingPointMatrix MatrixDerived, FloatingPointVector VectorDerived>
    requires SquareMatrix<MatrixDerived> && (MatrixDerived::RowsAtCompileTime == 4) &&
             VectorOfSize<VectorDerived, 3> &&
             std::same_as<typename MatrixDerived::Scalar, typename VectorDerived::Scalar>
[[nodiscard]] typename MatrixDerived::PlainObject
Rotate(const Eigen::MatrixBase<MatrixDerived>& matrix, typename MatrixDerived::Scalar angle,
       const Eigen::MatrixBase<VectorDerived>& axis)
{
    using Scalar = typename MatrixDerived::Scalar;
    if (axis.isZero())
    {
        return matrix;
    }
    return matrix * Eigen::Transform<Scalar, 3, Eigen::Isometry>(
                        Eigen::AngleAxis<Scalar>(angle, axis.normalized()))
                        .matrix();
}

// Returns matrix * S where S scales by the per-axis factors `scaling`. Works
// in any dimension D: `matrix` is a (D+1) x (D+1) affine transform of D-space
// and `scaling` is a D-vector.
template <FloatingPointMatrix MatrixDerived, FloatingPointVector VectorDerived>
    requires SquareMatrix<MatrixDerived> && (MatrixDerived::RowsAtCompileTime >= 3) &&
             (MatrixDerived::RowsAtCompileTime == VectorDerived::SizeAtCompileTime + 1) &&
             std::same_as<typename MatrixDerived::Scalar, typename VectorDerived::Scalar>
[[nodiscard]] typename MatrixDerived::PlainObject
Scale(const Eigen::MatrixBase<MatrixDerived>& matrix,
      const Eigen::MatrixBase<VectorDerived>& scaling)
{
    using Scalar = typename MatrixDerived::Scalar;
    constexpr int dim = MatrixDerived::RowsAtCompileTime - 1;
    return matrix *
           Eigen::Transform<Scalar, dim, Eigen::Affine>(Eigen::Scaling(scaling)).matrix();
}

} // namespace rad
