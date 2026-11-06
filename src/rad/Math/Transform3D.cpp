#include <rad/Math/Transform3D.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <system_error>
#include <utility>

namespace rad
{
namespace
{

enum class Handedness
{
    Right,
    Left
};

template <typename Value>
Value Require(Result<Value> result, const char* operation)
{
    if (!result)
    {
        throw std::system_error(result.error(), operation);
    }
    return std::move(result).value();
}

template <typename Value>
Result<Value> Finite(Value value)
{
    if (!value.allFinite())
    {
        return Failure(std::errc::result_out_of_range);
    }
    return Success(std::move(value));
}

template <typename T>
Result<Vector3<T>> UnitVector(const Vector3<T>& vector)
{
    if (!vector.allFinite())
    {
        return Failure(std::errc::invalid_argument);
    }
    const T scale = vector.cwiseAbs().maxCoeff();
    if (scale == T{0})
    {
        return Failure(std::errc::argument_out_of_domain);
    }
    // Scaling first avoids overflow/underflow when normalizing finite nonzero inputs.
    const Vector3<T> scaled = vector / scale;
    return Success(Vector3<T>(scaled / scaled.norm()));
}

template <typename T>
Result<Quaternion<T>> UnitQuaternion(const Quaternion<T>& quaternion)
{
    if (!quaternion.coeffs().allFinite())
    {
        return Failure(std::errc::invalid_argument);
    }
    const T scale = quaternion.coeffs().cwiseAbs().maxCoeff();
    if (scale == T{0})
    {
        return Failure(std::errc::argument_out_of_domain);
    }
    Quaternion<T> result;
    result.coeffs() = quaternion.coeffs() / scale;
    result.normalize();
    return Success(std::move(result));
}

template <typename T>
bool IsUnit(const Quaternion<T>& quaternion)
{
    return quaternion.coeffs().allFinite() &&
           std::abs(quaternion.squaredNorm() - T{1}) <= T{64} * std::numeric_limits<T>::epsilon();
}

template <typename T>
bool IsAffine(const Matrix4<T>& transform)
{
    const Vector4<T> affineRow{T{0}, T{0}, T{0}, T{1}};
    return transform.allFinite() &&
           (transform.row(3).transpose() - affineRow).cwiseAbs().maxCoeff() <=
               T{16} * std::numeric_limits<T>::epsilon();
}

bool ValidDepth(ClipDepth depth)
{
    return depth == ClipDepth::ZeroToOne || depth == ClipDepth::NegativeOneToOne;
}

template <typename T>
Result<Quaternion<T>> AxisAngle(T angle, const Vector3<T>& axis)
{
    if (!std::isfinite(angle))
    {
        return Failure(std::errc::invalid_argument);
    }
    auto unit = UnitVector(axis);
    if (!unit)
    {
        return Failure(unit.error());
    }
    return Success(Quaternion<T>(Eigen::AngleAxis<T>(angle, unit.value())));
}

template <typename T>
Result<Quaternion<T>> FromMatrix(const Matrix3<T>& rotation)
{
    const T tolerance = T{64} * std::numeric_limits<T>::epsilon();
    if (!rotation.allFinite())
    {
        return Failure(std::errc::invalid_argument);
    }
    const Matrix3<T> orthogonality = rotation.transpose() * rotation - Matrix3<T>::Identity();
    const T determinant = rotation.determinant();
    if (!orthogonality.allFinite() || !std::isfinite(determinant) ||
        orthogonality.cwiseAbs().maxCoeff() > tolerance || std::abs(determinant - T{1}) > tolerance)
    {
        return Failure(std::errc::argument_out_of_domain);
    }
    return UnitQuaternion(Quaternion<T>(rotation));
}

template <typename T>
Result<Quaternion<T>> Between(const Vector3<T>& from, const Vector3<T>& to)
{
    auto a = UnitVector(from);
    auto b = UnitVector(to);
    if (!a)
    {
        return Failure(a.error());
    }
    if (!b)
    {
        return Failure(b.error());
    }
    const Vector3<T> cross = a.value().cross(b.value());
    const T sine = cross.stableNorm();
    const T cosine = std::clamp(a.value().dot(b.value()), T{-1}, T{1});
    if (sine == T{0})
    {
        if (cosine >= T{0})
        {
            return Success(Quaternion<T>::Identity());
        }
        // Antiparallel directions have infinitely many rotation axes; choose a stable one.
        const Vector3<T> axis = a.value().unitOrthogonal();
        return AxisAngle(Pi<T>, axis);
    }
    return AxisAngle(std::atan2(sine, cosine), cross);
}

template <typename T>
Result<Quaternion<T>> Interpolate(const Quaternion<T>& from, const Quaternion<T>& to, T t,
                                  bool spherical)
{
    if (!IsUnit(from) || !IsUnit(to) || !std::isfinite(t) || t < T{0} || t > T{1})
    {
        return Failure(std::errc::invalid_argument);
    }
    if (spherical)
    {
        return UnitQuaternion(Quaternion<T>(from.slerp(t, to)));
    }
    Quaternion<T> end = to;
    if (from.dot(to) < T{0})
    {
        end.coeffs() = -end.coeffs();
    }
    Quaternion<T> result;
    result.coeffs() = (T{1} - t) * from.coeffs() + t * end.coeffs();
    return UnitQuaternion(result);
}

template <typename T>
Result<Matrix3<T>> Rotation3(const Quaternion<T>& rotation)
{
    if (!IsUnit(rotation))
    {
        return Failure(std::errc::invalid_argument);
    }
    return Success(Matrix3<T>(rotation.toRotationMatrix()));
}

template <typename T>
Result<Matrix4<T>> TRS(const Vector3<T>& translation, const Quaternion<T>& rotation,
                       const Vector3<T>& scale)
{
    if (!translation.allFinite() || !scale.allFinite())
    {
        return Failure(std::errc::invalid_argument);
    }
    auto linear = Rotation3(rotation);
    if (!linear)
    {
        return Failure(linear.error());
    }
    Matrix4<T> result = Matrix4<T>::Identity();
    result.template topLeftCorner<3, 3>() = linear.value() * scale.asDiagonal();
    result.template topRightCorner<3, 1>() = translation;
    return Finite(std::move(result));
}

template <typename T>
Result<Vector3<T>> AffineVector(const Matrix4<T>& transform, const Vector3<T>& vector, bool point)
{
    if (!IsAffine(transform) || !vector.allFinite())
    {
        return Failure(std::errc::invalid_argument);
    }
    Vector3<T> result = transform.template topLeftCorner<3, 3>() * vector;
    if (point)
    {
        result += transform.template topRightCorner<3, 1>();
    }
    return Finite(std::move(result));
}

template <typename T, int Size>
Result<MatrixMxN<T, Size, Size>> CheckedInverse(const MatrixMxN<T, Size, Size>& matrix)
{
    if (!matrix.allFinite())
    {
        return Failure(std::errc::invalid_argument);
    }
    // Normalize the matrix so the rank test is independent of its overall magnitude.
    const T scale = matrix.cwiseAbs().maxCoeff();
    if (scale == T{0})
    {
        return Failure(std::errc::argument_out_of_domain);
    }
    const MatrixMxN<T, Size, Size> scaled = matrix / scale;
    const Eigen::FullPivLU<MatrixMxN<T, Size, Size>> lu(scaled);
    if (!lu.isInvertible())
    {
        return Failure(std::errc::argument_out_of_domain);
    }
    return Finite(MatrixMxN<T, Size, Size>(lu.inverse() / scale));
}

template <typename T>
Result<Matrix3<T>> Normal(const Matrix4<T>& transform)
{
    if (!IsAffine(transform))
    {
        return Failure(std::errc::invalid_argument);
    }
    const Matrix3<T> linear = transform.template topLeftCorner<3, 3>();
    auto inverse = CheckedInverse<T, 3>(linear);
    if (!inverse)
    {
        return Failure(inverse.error());
    }
    return Success(Matrix3<T>(inverse.value().transpose()));
}

template <typename T>
Result<Matrix4<T>> View(const Vector3<T>& eye, const Vector3<T>& target, const Vector3<T>& up,
                        Handedness handedness)
{
    if (!eye.allFinite() || !target.allFinite())
    {
        return Failure(std::errc::invalid_argument);
    }
    const Vector3<T> delta = target - eye;
    if (!delta.allFinite())
    {
        return Failure(std::errc::result_out_of_range);
    }
    auto forward = UnitVector(delta);
    auto unitUp = UnitVector(up);
    if (!forward)
    {
        return Failure(forward.error());
    }
    if (!unitUp)
    {
        return Failure(unitUp.error());
    }
    const Vector3<T> viewZ =
        handedness == Handedness::Left ? forward.value() : Vector3<T>(-forward.value());
    const Vector3<T> cross = unitUp.value().cross(viewZ);
    if (cross.norm() <= T{16} * std::numeric_limits<T>::epsilon())
    {
        return Failure(std::errc::argument_out_of_domain);
    }
    const Vector3<T> right = cross.normalized();
    const Vector3<T> correctedUp = viewZ.cross(right);
    Matrix4<T> result = Matrix4<T>::Identity();
    result.template block<1, 3>(0, 0) = right.transpose();
    result.template block<1, 3>(1, 0) = correctedUp.transpose();
    result.template block<1, 3>(2, 0) = viewZ.transpose();
    result(0, 3) = -right.dot(eye);
    result(1, 3) = -correctedUp.dot(eye);
    result(2, 3) = -viewZ.dot(eye);
    return Finite(std::move(result));
}

template <typename T>
bool ValidPlanes(T zNear, T zFar, ClipDepth depth, bool perspective)
{
    return std::isfinite(zNear) && std::isfinite(zFar) &&
           (perspective ? zNear > T{0} : zNear >= T{0}) && zFar > zNear && ValidDepth(depth);
}

template <typename T>
void PerspectiveDepth(Matrix4<T>& result, T zNear, T zFar, ClipDepth depth)
{
    const T ratio = zNear / zFar;
    const T denominator = T{1} - ratio;
    const bool zeroToOne = depth == ClipDepth::ZeroToOne;
    result(2, 2) = -(zeroToOne ? T{1} : T{1} + ratio) / denominator;
    result(2, 3) = -(zeroToOne ? zNear : T{2} * zNear) / denominator;
    result(3, 2) = -T{1};
}

template <typename T>
Result<Matrix4<T>> FiniteProjection(Matrix4<T> result, Handedness handedness)
{
    if (handedness == Handedness::Left)
    {
        // P_LH = P_RH * diag(1, 1, -1, 1): reverse view-space Z, not clip-space depth.
        result.col(2) = -result.col(2);
    }
    if (result(0, 0) == T{0} || result(1, 1) == T{0} || result(2, 2) == T{0} ||
        (result(3, 3) == T{0} && result(2, 3) == T{0}))
    {
        return Failure(std::errc::result_out_of_range);
    }
    return Finite(std::move(result));
}

template <typename T>
Result<Matrix4<T>> Perspective(T fovY, T aspect, T zNear, T zFar, ClipDepth depth,
                               Handedness handedness)
{
    if (!ValidPlanes(zNear, zFar, depth, true) || !std::isfinite(fovY) || !std::isfinite(aspect) ||
        fovY <= T{0} || fovY >= Pi<T> || aspect <= T{0})
    {
        return Failure(std::errc::invalid_argument);
    }
    const T cotangent = T{1} / std::tan(fovY / T{2});
    Matrix4<T> result = Matrix4<T>::Zero();
    result(0, 0) = cotangent / aspect;
    result(1, 1) = cotangent;
    PerspectiveDepth(result, zNear, zFar, depth);
    return FiniteProjection(std::move(result), handedness);
}

template <typename T>
Result<Matrix4<T>> BoxProjection(T left, T right, T bottom, T top, T zNear, T zFar, ClipDepth depth,
                                 bool perspective, Handedness handedness)
{
    if (!ValidPlanes(zNear, zFar, depth, perspective) || !std::isfinite(left) ||
        !std::isfinite(right) || !std::isfinite(bottom) || !std::isfinite(top) || left >= right ||
        bottom >= top)
    {
        return Failure(std::errc::invalid_argument);
    }
    const T width = right - left;
    const T height = top - bottom;
    if (!std::isfinite(width) || !std::isfinite(height))
    {
        return Failure(std::errc::result_out_of_range);
    }
    Matrix4<T> result = Matrix4<T>::Zero();
    const bool zeroToOne = depth == ClipDepth::ZeroToOne;
    if (perspective)
    {
        result(0, 0) = T{2} * (zNear / width);
        result(1, 1) = T{2} * (zNear / height);
        result(0, 2) = right / width + left / width;
        result(1, 2) = top / height + bottom / height;
        PerspectiveDepth(result, zNear, zFar, depth);
    }
    else
    {
        const T distance = zFar - zNear;
        result(0, 0) = T{2} / width;
        result(1, 1) = T{2} / height;
        result(0, 3) = -(right / width + left / width);
        result(1, 3) = -(top / height + bottom / height);
        result(2, 2) = -(zeroToOne ? T{1} : T{2}) / distance;
        result(2, 3) = zeroToOne ? -zNear / distance : -(zFar / distance + zNear / distance);
        result(3, 3) = T{1};
    }
    return FiniteProjection(std::move(result), handedness);
}

template <typename T>
Result<Vector3<T>> HomogeneousPoint(const Matrix4<T>& transform, const Vector3<T>& point)
{
    if (!transform.allFinite() || !point.allFinite())
    {
        return Failure(std::errc::invalid_argument);
    }
    const Vector4<T> homogeneous = transform * Vector4<T>{point.x(), point.y(), point.z(), T{1}};
    if (!homogeneous.allFinite())
    {
        return Failure(std::errc::result_out_of_range);
    }
    if (homogeneous.w() == T{0})
    {
        return Failure(std::errc::argument_out_of_domain);
    }
    return Finite(Vector3<T>(homogeneous.template head<3>() / homogeneous.w()));
}

template <typename T>
bool ValidViewport(const Viewport<T>& viewport, ClipDepth depth)
{
    return std::isfinite(viewport.x) && std::isfinite(viewport.y) &&
           std::isfinite(viewport.width) && std::isfinite(viewport.height) &&
           std::isfinite(viewport.minDepth) && std::isfinite(viewport.maxDepth) &&
           viewport.width > T{0} && viewport.height != T{0} &&
           viewport.minDepth < viewport.maxDepth &&
           std::isfinite(viewport.maxDepth - viewport.minDepth) && ValidDepth(depth);
}

template <typename T>
Result<Vector3<T>> WindowPoint(const Matrix4<T>& transform, const Vector3<T>& point,
                               const Viewport<T>& viewport, ClipDepth depth, bool unproject)
{
    if (!ValidViewport(viewport, depth) || !point.allFinite())
    {
        return Failure(std::errc::invalid_argument);
    }
    const T depthSize = viewport.maxDepth - viewport.minDepth;
    const bool zeroToOne = depth == ClipDepth::ZeroToOne;
    Vector3<T> result;
    if (unproject)
    {
        result.x() = T{2} * ((point.x() - viewport.x) / viewport.width) - T{1};
        result.y() = T{2} * ((point.y() - viewport.y) / viewport.height) - T{1};
        const T z = (point.z() - viewport.minDepth) / depthSize;
        result.z() = zeroToOne ? z : T{2} * z - T{1};
        if (!result.allFinite())
        {
            return Failure(std::errc::result_out_of_range);
        }
        return HomogeneousPoint(transform, result);
    }
    auto ndc = HomogeneousPoint(transform, point);
    if (!ndc)
    {
        return Failure(ndc.error());
    }
    result.x() = viewport.x + (ndc.value().x() / T{2} + T{0.5}) * viewport.width;
    result.y() = viewport.y + (ndc.value().y() / T{2} + T{0.5}) * viewport.height;
    const T z = zeroToOne ? ndc.value().z() : ndc.value().z() / T{2} + T{0.5};
    result.z() = viewport.minDepth + z * depthSize;
    return Finite(std::move(result));
}

} // namespace

Quaternionf QuaternionFromAxisAngle(float angle, const Vector3f& axis)
{
    return Require(AxisAngle(angle, axis), "QuaternionFromAxisAngle");
}

Quaterniond QuaternionFromAxisAngle(double angle, const Vector3d& axis)
{
    return Require(AxisAngle(angle, axis), "QuaternionFromAxisAngle");
}

Quaternionf QuaternionFromRotationMatrix(const Matrix3f& rotation)
{
    return Require(FromMatrix(rotation), "QuaternionFromRotationMatrix");
}

Quaterniond QuaternionFromRotationMatrix(const Matrix3d& rotation)
{
    return Require(FromMatrix(rotation), "QuaternionFromRotationMatrix");
}

Quaternionf RotationBetween(const Vector3f& from, const Vector3f& to)
{
    return Require(Between(from, to), "RotationBetween");
}

Quaterniond RotationBetween(const Vector3d& from, const Vector3d& to)
{
    return Require(Between(from, to), "RotationBetween");
}

Quaternionf Normalize(const Quaternionf& quaternion)
{
    return Require(UnitQuaternion(quaternion), "Normalize quaternion");
}

Quaterniond Normalize(const Quaterniond& quaternion)
{
    return Require(UnitQuaternion(quaternion), "Normalize quaternion");
}

Quaternionf Conjugate(const Quaternionf& quaternion)
{
    if (!quaternion.coeffs().allFinite())
    {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument), "Conjugate");
    }
    return quaternion.conjugate();
}

Quaterniond Conjugate(const Quaterniond& quaternion)
{
    if (!quaternion.coeffs().allFinite())
    {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument), "Conjugate");
    }
    return quaternion.conjugate();
}

Quaternionf Inverse(const Quaternionf& quaternion)
{
    if (!IsUnit(quaternion))
    {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument),
                                "Inverse quaternion");
    }
    return quaternion.conjugate();
}

Quaterniond Inverse(const Quaterniond& quaternion)
{
    if (!IsUnit(quaternion))
    {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument),
                                "Inverse quaternion");
    }
    return quaternion.conjugate();
}

Quaternionf Nlerp(const Quaternionf& from, const Quaternionf& to, float t)
{
    return Require(Interpolate(from, to, t, false), "Nlerp");
}

Quaterniond Nlerp(const Quaterniond& from, const Quaterniond& to, double t)
{
    return Require(Interpolate(from, to, t, false), "Nlerp");
}

Quaternionf Slerp(const Quaternionf& from, const Quaternionf& to, float t)
{
    return Require(Interpolate(from, to, t, true), "Slerp");
}

Quaterniond Slerp(const Quaterniond& from, const Quaterniond& to, double t)
{
    return Require(Interpolate(from, to, t, true), "Slerp");
}

Matrix3f ToRotationMatrix3(const Quaternionf& rotation)
{
    return Require(Rotation3(rotation), "ToRotationMatrix3");
}

Matrix3d ToRotationMatrix3(const Quaterniond& rotation)
{
    return Require(Rotation3(rotation), "ToRotationMatrix3");
}

Matrix4f ToRotationMatrix4(const Quaternionf& rotation)
{
    return ComposeTRS(Vector3f::Zero(), rotation, Vector3f::Ones());
}

Matrix4d ToRotationMatrix4(const Quaterniond& rotation)
{
    return ComposeTRS(Vector3d::Zero(), rotation, Vector3d::Ones());
}

Vector3f RotateVector(const Quaternionf& rotation, const Vector3f& vector)
{
    if (!IsUnit(rotation) || !vector.allFinite())
    {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument), "RotateVector");
    }
    return Require(Finite(Vector3f(rotation * vector)), "RotateVector");
}

Vector3d RotateVector(const Quaterniond& rotation, const Vector3d& vector)
{
    if (!IsUnit(rotation) || !vector.allFinite())
    {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument), "RotateVector");
    }
    return Require(Finite(Vector3d(rotation * vector)), "RotateVector");
}

Matrix4f TranslationMatrix(const Vector3f& translation)
{
    return ComposeTRS(translation, Quaternionf::Identity(), Vector3f::Ones());
}

Matrix4d TranslationMatrix(const Vector3d& translation)
{
    return ComposeTRS(translation, Quaterniond::Identity(), Vector3d::Ones());
}

Matrix4f ScaleMatrix(const Vector3f& scale)
{
    return ComposeTRS(Vector3f::Zero(), Quaternionf::Identity(), scale);
}

Matrix4d ScaleMatrix(const Vector3d& scale)
{
    return ComposeTRS(Vector3d::Zero(), Quaterniond::Identity(), scale);
}

Matrix4f ComposeTRS(const Vector3f& translation, const Quaternionf& rotation, const Vector3f& scale)
{
    return Require(TRS(translation, rotation, scale), "ComposeTRS");
}

Matrix4d ComposeTRS(const Vector3d& translation, const Quaterniond& rotation, const Vector3d& scale)
{
    return Require(TRS(translation, rotation, scale), "ComposeTRS");
}

Vector3f TransformPoint(const Matrix4f& transform, const Vector3f& point)
{
    return Require(AffineVector(transform, point, true), "TransformPoint");
}

Vector3d TransformPoint(const Matrix4d& transform, const Vector3d& point)
{
    return Require(AffineVector(transform, point, true), "TransformPoint");
}

Vector3f TransformDirection(const Matrix4f& transform, const Vector3f& direction)
{
    return Require(AffineVector(transform, direction, false), "TransformDirection");
}

Vector3d TransformDirection(const Matrix4d& transform, const Vector3d& direction)
{
    return Require(AffineVector(transform, direction, false), "TransformDirection");
}

Matrix3f NormalMatrix(const Matrix4f& transform)
{
    return Require(TryNormalMatrix(transform), "NormalMatrix");
}

Matrix3d NormalMatrix(const Matrix4d& transform)
{
    return Require(TryNormalMatrix(transform), "NormalMatrix");
}

Result<Matrix3f> TryNormalMatrix(const Matrix4f& transform)
{
    return Normal(transform);
}

Result<Matrix3d> TryNormalMatrix(const Matrix4d& transform)
{
    return Normal(transform);
}

Vector3f TransformNormal(const Matrix4f& transform, const Vector3f& normal)
{
    if (!normal.allFinite())
    {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument),
                                "TransformNormal");
    }
    return Require(Finite(Vector3f(NormalMatrix(transform) * normal)), "TransformNormal");
}

Vector3d TransformNormal(const Matrix4d& transform, const Vector3d& normal)
{
    if (!normal.allFinite())
    {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument),
                                "TransformNormal");
    }
    return Require(Finite(Vector3d(NormalMatrix(transform) * normal)), "TransformNormal");
}

Result<Matrix4f> TryInverse(const Matrix4f& matrix)
{
    return CheckedInverse<float, 4>(matrix);
}

Result<Matrix4d> TryInverse(const Matrix4d& matrix)
{
    return CheckedInverse<double, 4>(matrix);
}

Matrix4f LookAtRH(const Vector3f& eye, const Vector3f& target, const Vector3f& up)
{
    return Require(View(eye, target, up, Handedness::Right), "LookAtRH");
}

Matrix4d LookAtRH(const Vector3d& eye, const Vector3d& target, const Vector3d& up)
{
    return Require(View(eye, target, up, Handedness::Right), "LookAtRH");
}

Matrix4f LookAtLH(const Vector3f& eye, const Vector3f& target, const Vector3f& up)
{
    return Require(View(eye, target, up, Handedness::Left), "LookAtLH");
}

Matrix4d LookAtLH(const Vector3d& eye, const Vector3d& target, const Vector3d& up)
{
    return Require(View(eye, target, up, Handedness::Left), "LookAtLH");
}

Matrix4f PerspectiveRH(float fovY, float aspect, float zNear, float zFar, ClipDepth depth)
{
    return Require(Perspective(fovY, aspect, zNear, zFar, depth, Handedness::Right),
                   "PerspectiveRH");
}

Matrix4d PerspectiveRH(double fovY, double aspect, double zNear, double zFar, ClipDepth depth)
{
    return Require(Perspective(fovY, aspect, zNear, zFar, depth, Handedness::Right),
                   "PerspectiveRH");
}

Matrix4f PerspectiveLH(float fovY, float aspect, float zNear, float zFar, ClipDepth depth)
{
    return Require(Perspective(fovY, aspect, zNear, zFar, depth, Handedness::Left),
                   "PerspectiveLH");
}

Matrix4d PerspectiveLH(double fovY, double aspect, double zNear, double zFar, ClipDepth depth)
{
    return Require(Perspective(fovY, aspect, zNear, zFar, depth, Handedness::Left),
                   "PerspectiveLH");
}

Matrix4f FrustumRH(float left, float right, float bottom, float top, float zNear, float zFar,
                   ClipDepth depth)
{
    return Require(
        BoxProjection(left, right, bottom, top, zNear, zFar, depth, true, Handedness::Right),
        "FrustumRH");
}

Matrix4d FrustumRH(double left, double right, double bottom, double top, double zNear, double zFar,
                   ClipDepth depth)
{
    return Require(
        BoxProjection(left, right, bottom, top, zNear, zFar, depth, true, Handedness::Right),
        "FrustumRH");
}

Matrix4f FrustumLH(float left, float right, float bottom, float top, float zNear, float zFar,
                   ClipDepth depth)
{
    return Require(
        BoxProjection(left, right, bottom, top, zNear, zFar, depth, true, Handedness::Left),
        "FrustumLH");
}

Matrix4d FrustumLH(double left, double right, double bottom, double top, double zNear, double zFar,
                   ClipDepth depth)
{
    return Require(
        BoxProjection(left, right, bottom, top, zNear, zFar, depth, true, Handedness::Left),
        "FrustumLH");
}

Matrix4f OrthographicRH(float left, float right, float bottom, float top, float zNear, float zFar,
                        ClipDepth depth)
{
    return Require(
        BoxProjection(left, right, bottom, top, zNear, zFar, depth, false, Handedness::Right),
        "OrthographicRH");
}

Matrix4d OrthographicRH(double left, double right, double bottom, double top, double zNear,
                        double zFar, ClipDepth depth)
{
    return Require(
        BoxProjection(left, right, bottom, top, zNear, zFar, depth, false, Handedness::Right),
        "OrthographicRH");
}

Matrix4f OrthographicLH(float left, float right, float bottom, float top, float zNear, float zFar,
                        ClipDepth depth)
{
    return Require(
        BoxProjection(left, right, bottom, top, zNear, zFar, depth, false, Handedness::Left),
        "OrthographicLH");
}

Matrix4d OrthographicLH(double left, double right, double bottom, double top, double zNear,
                        double zFar, ClipDepth depth)
{
    return Require(
        BoxProjection(left, right, bottom, top, zNear, zFar, depth, false, Handedness::Left),
        "OrthographicLH");
}

Vector3f ProjectToNdc(const Matrix4f& worldToClip, const Vector3f& point)
{
    return Require(TryProjectToNdc(worldToClip, point), "ProjectToNdc");
}

Vector3d ProjectToNdc(const Matrix4d& worldToClip, const Vector3d& point)
{
    return Require(TryProjectToNdc(worldToClip, point), "ProjectToNdc");
}

Vector3f UnprojectFromNdc(const Matrix4f& clipToWorld, const Vector3f& ndc)
{
    return Require(TryUnprojectFromNdc(clipToWorld, ndc), "UnprojectFromNdc");
}

Vector3d UnprojectFromNdc(const Matrix4d& clipToWorld, const Vector3d& ndc)
{
    return Require(TryUnprojectFromNdc(clipToWorld, ndc), "UnprojectFromNdc");
}

Result<Vector3f> TryProjectToNdc(const Matrix4f& worldToClip, const Vector3f& point)
{
    return HomogeneousPoint(worldToClip, point);
}

Result<Vector3d> TryProjectToNdc(const Matrix4d& worldToClip, const Vector3d& point)
{
    return HomogeneousPoint(worldToClip, point);
}

Result<Vector3f> TryUnprojectFromNdc(const Matrix4f& clipToWorld, const Vector3f& ndc)
{
    return HomogeneousPoint(clipToWorld, ndc);
}

Result<Vector3d> TryUnprojectFromNdc(const Matrix4d& clipToWorld, const Vector3d& ndc)
{
    return HomogeneousPoint(clipToWorld, ndc);
}

Vector3f ProjectToViewport(const Matrix4f& worldToClip, const Vector3f& point,
                           const Viewportf& viewport, ClipDepth depth)
{
    return Require(WindowPoint(worldToClip, point, viewport, depth, false), "ProjectToViewport");
}

Vector3d ProjectToViewport(const Matrix4d& worldToClip, const Vector3d& point,
                           const Viewportd& viewport, ClipDepth depth)
{
    return Require(WindowPoint(worldToClip, point, viewport, depth, false), "ProjectToViewport");
}

Vector3f UnprojectFromViewport(const Matrix4f& clipToWorld, const Vector3f& point,
                               const Viewportf& viewport, ClipDepth depth)
{
    return Require(WindowPoint(clipToWorld, point, viewport, depth, true), "UnprojectFromViewport");
}

Vector3d UnprojectFromViewport(const Matrix4d& clipToWorld, const Vector3d& point,
                               const Viewportd& viewport, ClipDepth depth)
{
    return Require(WindowPoint(clipToWorld, point, viewport, depth, true), "UnprojectFromViewport");
}

} // namespace rad
