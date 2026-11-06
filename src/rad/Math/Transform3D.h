#pragma once

#include <rad/Math/Matrix.h>
#include <rad/Core/Result.h>

namespace rad
{

// 3D transformations use right-handed rotations, column vectors and radians.
// clip = projection * view * model * point.
// RH view/projection functions look along local -Z; LH variants look along local +Z.
// Choosing LH does not change quaternion rotations, cross products or triangle winding.
// Quaternion products qB * qA apply A first. Rotation inputs must be unit quaternions.
// Eigen quaternion constructors take (w, x, y, z); coeffs() stores (x, y, z, w).
// All functions return owning values. Invalid or unrepresentable inputs throw std::system_error;
// Try functions instead return invalid_argument, argument_out_of_domain or result_out_of_range.
template <typename T>
using Quaternion = Eigen::Quaternion<T>;

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

enum class ClipDepth
{
    ZeroToOne,
    NegativeOneToOne
};

// x/y are the window coordinates of NDC (-1, -1).
// Positive height preserves the NDC Y direction; negative height reverses it.
// For NDC +Y pointing up on a top-left screen: y = top + h, height = -h.
template <typename T>
struct Viewport
{
    T x = 0;
    T y = 0;
    T width = 1;
    T height = 1;
    T minDepth = 0;
    T maxDepth = 1;
};

using Viewportf = Viewport<float>;
using Viewportd = Viewport<double>;

////////////////////////////////////////////////////////////////////////////////
// Quaternions
////////////////////////////////////////////////////////////////////////////////

// Axis and direction inputs may have any nonzero length.
[[nodiscard]] Quaternionf QuaternionFromAxisAngle(float angle, const Vector3f& axis);
[[nodiscard]] Quaterniond QuaternionFromAxisAngle(double angle, const Vector3d& axis);
[[nodiscard]] Quaternionf QuaternionFromRotationMatrix(const Matrix3f& rotation);
[[nodiscard]] Quaterniond QuaternionFromRotationMatrix(const Matrix3d& rotation);
[[nodiscard]] Quaternionf RotationBetween(const Vector3f& from, const Vector3f& to);
[[nodiscard]] Quaterniond RotationBetween(const Vector3d& from, const Vector3d& to);

[[nodiscard]] Quaternionf Normalize(const Quaternionf& quaternion);
[[nodiscard]] Quaterniond Normalize(const Quaterniond& quaternion);
[[nodiscard]] Quaternionf Conjugate(const Quaternionf& quaternion);
[[nodiscard]] Quaterniond Conjugate(const Quaterniond& quaternion);
// Inverse is for unit rotation quaternions.
[[nodiscard]] Quaternionf Inverse(const Quaternionf& quaternion);
[[nodiscard]] Quaterniond Inverse(const Quaterniond& quaternion);

// Shortest-path interpolation; t must be in [0, 1]. Results are normalized.
[[nodiscard]] Quaternionf Nlerp(const Quaternionf& from, const Quaternionf& to, float t);
[[nodiscard]] Quaterniond Nlerp(const Quaterniond& from, const Quaterniond& to, double t);
[[nodiscard]] Quaternionf Slerp(const Quaternionf& from, const Quaternionf& to, float t);
[[nodiscard]] Quaterniond Slerp(const Quaterniond& from, const Quaterniond& to, double t);

[[nodiscard]] Matrix3f ToRotationMatrix3(const Quaternionf& rotation);
[[nodiscard]] Matrix3d ToRotationMatrix3(const Quaterniond& rotation);
[[nodiscard]] Matrix4f ToRotationMatrix4(const Quaternionf& rotation);
[[nodiscard]] Matrix4d ToRotationMatrix4(const Quaterniond& rotation);
[[nodiscard]] Vector3f RotateVector(const Quaternionf& rotation, const Vector3f& vector);
[[nodiscard]] Vector3d RotateVector(const Quaterniond& rotation, const Vector3d& vector);

////////////////////////////////////////////////////////////////////////////////
// Affine transformations
////////////////////////////////////////////////////////////////////////////////

[[nodiscard]] Matrix4f TranslationMatrix(const Vector3f& translation);
[[nodiscard]] Matrix4d TranslationMatrix(const Vector3d& translation);
[[nodiscard]] Matrix4f ScaleMatrix(const Vector3f& scale);
[[nodiscard]] Matrix4d ScaleMatrix(const Vector3d& scale);
// Returns T * R * S, including negative or zero scale factors.
[[nodiscard]] Matrix4f ComposeTRS(const Vector3f& translation, const Quaternionf& rotation,
                                  const Vector3f& scale);
[[nodiscard]] Matrix4d ComposeTRS(const Vector3d& translation, const Quaterniond& rotation,
                                  const Vector3d& scale);

// These accept affine matrices only. Points use w = 1, directions use w = 0.
[[nodiscard]] Vector3f TransformPoint(const Matrix4f& transform, const Vector3f& point);
[[nodiscard]] Vector3d TransformPoint(const Matrix4d& transform, const Vector3d& point);
[[nodiscard]] Vector3f TransformDirection(const Matrix4f& transform, const Vector3f& direction);
[[nodiscard]] Vector3d TransformDirection(const Matrix4d& transform, const Vector3d& direction);

// Inverse transpose of the linear 3x3 block. Normals are NOT normalized automatically.
// For batches, compute NormalMatrix once and multiply each normal by it.
[[nodiscard]] Matrix3f NormalMatrix(const Matrix4f& transform);
[[nodiscard]] Matrix3d NormalMatrix(const Matrix4d& transform);
[[nodiscard]] Result<Matrix3f> TryNormalMatrix(const Matrix4f& transform);
[[nodiscard]] Result<Matrix3d> TryNormalMatrix(const Matrix4d& transform);
[[nodiscard]] Vector3f TransformNormal(const Matrix4f& transform, const Vector3f& normal);
[[nodiscard]] Vector3d TransformNormal(const Matrix4d& transform, const Vector3d& normal);
[[nodiscard]] Result<Matrix4f> TryInverse(const Matrix4f& matrix);
[[nodiscard]] Result<Matrix4d> TryInverse(const Matrix4d& matrix);

////////////////////////////////////////////////////////////////////////////////
// View and projection matrices
////////////////////////////////////////////////////////////////////////////////

// World -> view; the eye maps to the origin. RH looks along local -Z, LH along local +Z.
// up need not be normalized, but must not be parallel to target - eye.
[[nodiscard]] Matrix4f LookAtRH(const Vector3f& eye, const Vector3f& target, const Vector3f& up);
[[nodiscard]] Matrix4d LookAtRH(const Vector3d& eye, const Vector3d& target, const Vector3d& up);
[[nodiscard]] Matrix4f LookAtLH(const Vector3f& eye, const Vector3f& target, const Vector3f& up);
[[nodiscard]] Matrix4d LookAtLH(const Vector3d& eye, const Vector3d& target, const Vector3d& up);

// fovY is the vertical field of view in (0, Pi). aspect > 0; 0 < zNear < zFar.
// Depth range is explicit; no implicit Vulkan Y flip or reversed-Z.
[[nodiscard]] Matrix4f PerspectiveRH(float fovY, float aspect, float zNear, float zFar,
                                     ClipDepth depth);
[[nodiscard]] Matrix4d PerspectiveRH(double fovY, double aspect, double zNear, double zFar,
                                     ClipDepth depth);
[[nodiscard]] Matrix4f PerspectiveLH(float fovY, float aspect, float zNear, float zFar,
                                     ClipDepth depth);
[[nodiscard]] Matrix4d PerspectiveLH(double fovY, double aspect, double zNear, double zFar,
                                     ClipDepth depth);
// Off-center perspective bounds on the near plane: left < right, bottom < top.
[[nodiscard]] Matrix4f FrustumRH(float left, float right, float bottom, float top, float zNear,
                                 float zFar, ClipDepth depth);
[[nodiscard]] Matrix4d FrustumRH(double left, double right, double bottom, double top, double zNear,
                                 double zFar, ClipDepth depth);
[[nodiscard]] Matrix4f FrustumLH(float left, float right, float bottom, float top, float zNear,
                                 float zFar, ClipDepth depth);
[[nodiscard]] Matrix4d FrustumLH(double left, double right, double bottom, double top, double zNear,
                                 double zFar, ClipDepth depth);
// Orthographic distances satisfy 0 <= zNear < zFar.
[[nodiscard]] Matrix4f OrthographicRH(float left, float right, float bottom, float top, float zNear,
                                      float zFar, ClipDepth depth);
[[nodiscard]] Matrix4d OrthographicRH(double left, double right, double bottom, double top,
                                      double zNear, double zFar, ClipDepth depth);
[[nodiscard]] Matrix4f OrthographicLH(float left, float right, float bottom, float top, float zNear,
                                      float zFar, ClipDepth depth);
[[nodiscard]] Matrix4d OrthographicLH(double left, double right, double bottom, double top,
                                      double zNear, double zFar, ClipDepth depth);

////////////////////////////////////////////////////////////////////////////////
// Projection and unprojection
////////////////////////////////////////////////////////////////////////////////

// Includes the homogeneous divide. Does not clip: off-screen and behind-camera points are valid.
// w must be nonzero and the result representable. Unproject takes an already inverted matrix.
[[nodiscard]] Vector3f ProjectToNdc(const Matrix4f& worldToClip, const Vector3f& point);
[[nodiscard]] Vector3d ProjectToNdc(const Matrix4d& worldToClip, const Vector3d& point);
[[nodiscard]] Vector3f UnprojectFromNdc(const Matrix4f& clipToWorld, const Vector3f& ndc);
[[nodiscard]] Vector3d UnprojectFromNdc(const Matrix4d& clipToWorld, const Vector3d& ndc);
[[nodiscard]] Result<Vector3f> TryProjectToNdc(const Matrix4f& worldToClip, const Vector3f& point);
[[nodiscard]] Result<Vector3d> TryProjectToNdc(const Matrix4d& worldToClip, const Vector3d& point);
[[nodiscard]] Result<Vector3f> TryUnprojectFromNdc(const Matrix4f& clipToWorld,
                                                   const Vector3f& ndc);
[[nodiscard]] Result<Vector3d> TryUnprojectFromNdc(const Matrix4d& clipToWorld,
                                                   const Vector3d& ndc);

// Window coordinates are continuous (no half-pixel offset).
// width > 0, height != 0 and minDepth < maxDepth; results are not clamped.
// window.y = viewport.y + (ndc.y + 1) * viewport.height / 2.
[[nodiscard]] Vector3f ProjectToViewport(const Matrix4f& worldToClip, const Vector3f& point,
                                         const Viewportf& viewport, ClipDepth depth);
[[nodiscard]] Vector3d ProjectToViewport(const Matrix4d& worldToClip, const Vector3d& point,
                                         const Viewportd& viewport, ClipDepth depth);
[[nodiscard]] Vector3f UnprojectFromViewport(const Matrix4f& clipToWorld, const Vector3f& point,
                                             const Viewportf& viewport, ClipDepth depth);
[[nodiscard]] Vector3d UnprojectFromViewport(const Matrix4d& clipToWorld, const Vector3d& point,
                                             const Viewportd& viewport, ClipDepth depth);

////////////////////////////////////////////////////////////////////////////////
// Eigen expression adapters (evaluate once, then call the compiled overload)
////////////////////////////////////////////////////////////////////////////////

// Bind input types directly to avoid ambiguity when mixing concrete values and expressions.
template <FloatingPointMatrix Derived>
    requires SquareMatrix<Derived> && (Derived::RowsAtCompileTime == 3)
[[nodiscard]] auto QuaternionFromRotationMatrix(const Derived& rotation)
{
    return QuaternionFromRotationMatrix(Matrix3<typename Derived::Scalar>(rotation));
}

template <FloatingPointVector A, FloatingPointVector B>
    requires VectorOfSize<A, 3> && VectorOfSize<B, 3> &&
             std::same_as<typename A::Scalar, typename B::Scalar>
[[nodiscard]] auto RotationBetween(const A& from, const B& to)
{
    using V = Vector3<typename A::Scalar>;
    return RotationBetween(V(from), V(to));
}

template <FloatingPointVector Derived>
    requires VectorOfSize<Derived, 3>
[[nodiscard]] auto TranslationMatrix(const Derived& translation)
{
    return TranslationMatrix(Vector3<typename Derived::Scalar>(translation));
}

template <FloatingPointVector Derived>
    requires VectorOfSize<Derived, 3>
[[nodiscard]] auto ScaleMatrix(const Derived& scale)
{
    return ScaleMatrix(Vector3<typename Derived::Scalar>(scale));
}

template <FloatingPointMatrix M, FloatingPointVector V>
    requires SquareMatrix<M> && (M::RowsAtCompileTime == 4) &&
             VectorOfSize<V, 3> && std::same_as<typename M::Scalar, typename V::Scalar>
[[nodiscard]] auto TransformPoint(const M& transform, const V& point)
{
    using T = typename M::Scalar;
    return TransformPoint(Matrix4<T>(transform), Vector3<T>(point));
}

template <FloatingPointMatrix M, FloatingPointVector V>
    requires SquareMatrix<M> && (M::RowsAtCompileTime == 4) &&
             VectorOfSize<V, 3> && std::same_as<typename M::Scalar, typename V::Scalar>
[[nodiscard]] auto TransformDirection(const M& transform, const V& direction)
{
    using T = typename M::Scalar;
    return TransformDirection(Matrix4<T>(transform), Vector3<T>(direction));
}

template <FloatingPointMatrix M, FloatingPointVector V>
    requires SquareMatrix<M> && (M::RowsAtCompileTime == 4) &&
             VectorOfSize<V, 3> && std::same_as<typename M::Scalar, typename V::Scalar>
[[nodiscard]] auto TransformNormal(const M& transform, const V& normal)
{
    using T = typename M::Scalar;
    return TransformNormal(Matrix4<T>(transform), Vector3<T>(normal));
}

template <FloatingPointMatrix Derived>
    requires SquareMatrix<Derived> && (Derived::RowsAtCompileTime == 4)
[[nodiscard]] auto NormalMatrix(const Derived& transform)
{
    return NormalMatrix(Matrix4<typename Derived::Scalar>(transform));
}

template <FloatingPointMatrix Derived>
    requires SquareMatrix<Derived> && (Derived::RowsAtCompileTime == 4)
[[nodiscard]] auto TryNormalMatrix(const Derived& transform)
{
    return TryNormalMatrix(Matrix4<typename Derived::Scalar>(transform));
}

template <FloatingPointMatrix Derived>
    requires SquareMatrix<Derived> && (Derived::RowsAtCompileTime == 4)
[[nodiscard]] auto TryInverse(const Derived& matrix)
{
    return TryInverse(Matrix4<typename Derived::Scalar>(matrix));
}

template <FloatingPointVector E, FloatingPointVector C, FloatingPointVector U>
    requires VectorOfSize<E, 3> && VectorOfSize<C, 3> && VectorOfSize<U, 3> &&
             std::same_as<typename E::Scalar, typename C::Scalar> &&
             std::same_as<typename E::Scalar, typename U::Scalar>
[[nodiscard]] auto LookAtRH(const E& eye, const C& target, const U& up)
{
    using V = Vector3<typename E::Scalar>;
    return LookAtRH(V(eye), V(target), V(up));
}

template <FloatingPointVector E, FloatingPointVector C, FloatingPointVector U>
    requires VectorOfSize<E, 3> && VectorOfSize<C, 3> && VectorOfSize<U, 3> &&
             std::same_as<typename E::Scalar, typename C::Scalar> &&
             std::same_as<typename E::Scalar, typename U::Scalar>
[[nodiscard]] auto LookAtLH(const E& eye, const C& target, const U& up)
{
    using V = Vector3<typename E::Scalar>;
    return LookAtLH(V(eye), V(target), V(up));
}

template <FloatingPointMatrix M, FloatingPointVector V>
    requires SquareMatrix<M> && (M::RowsAtCompileTime == 4) &&
             VectorOfSize<V, 3> && std::same_as<typename M::Scalar, typename V::Scalar>
[[nodiscard]] auto ProjectToNdc(const M& worldToClip, const V& point)
{
    using T = typename M::Scalar;
    return ProjectToNdc(Matrix4<T>(worldToClip), Vector3<T>(point));
}

template <FloatingPointMatrix M, FloatingPointVector V>
    requires SquareMatrix<M> && (M::RowsAtCompileTime == 4) &&
             VectorOfSize<V, 3> && std::same_as<typename M::Scalar, typename V::Scalar>
[[nodiscard]] auto UnprojectFromNdc(const M& clipToWorld, const V& ndc)
{
    using T = typename M::Scalar;
    return UnprojectFromNdc(Matrix4<T>(clipToWorld), Vector3<T>(ndc));
}

template <FloatingPointMatrix M, FloatingPointVector V>
    requires SquareMatrix<M> && (M::RowsAtCompileTime == 4) &&
             VectorOfSize<V, 3> && std::same_as<typename M::Scalar, typename V::Scalar>
[[nodiscard]] auto TryProjectToNdc(const M& worldToClip, const V& point)
{
    using T = typename M::Scalar;
    return TryProjectToNdc(Matrix4<T>(worldToClip), Vector3<T>(point));
}

template <FloatingPointMatrix M, FloatingPointVector V>
    requires SquareMatrix<M> && (M::RowsAtCompileTime == 4) &&
             VectorOfSize<V, 3> && std::same_as<typename M::Scalar, typename V::Scalar>
[[nodiscard]] auto TryUnprojectFromNdc(const M& clipToWorld, const V& ndc)
{
    using T = typename M::Scalar;
    return TryUnprojectFromNdc(Matrix4<T>(clipToWorld), Vector3<T>(ndc));
}

} // namespace rad
