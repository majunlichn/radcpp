#include <rad/Math/Transform3D.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <system_error>
#include <type_traits>

namespace
{

template <typename T>
constexpr T Tolerance = std::is_same_v<T, float> ? T{0.00002} : T{0.000000000001};

template <typename T>
void TestQuaternions()
{
    using namespace rad;
    using V = Vector3<T>;
    using Q = Quaternion<T>;
    const V x{T{1}, T{0}, T{0}};
    const V y{T{0}, T{1}, T{0}};
    const V z{T{0}, T{0}, T{1}};
    const Q identity = Q::Identity();
    const Q rz = QuaternionFromAxisAngle(HalfPi<T>, V(z * T{5}));
    const Q rx = QuaternionFromAxisAngle(HalfPi<T>, x);

    EXPECT_NEAR(rz.w(), std::sqrt(T{0.5}), Tolerance<T>);
    EXPECT_NEAR(rz.coeffs()(2), std::sqrt(T{0.5}), Tolerance<T>);
    EXPECT_TRUE(RotateVector(rz, x).isApprox(y, Tolerance<T>));
    EXPECT_TRUE(RotateVector(rx, y).isApprox(z, Tolerance<T>));
    const Q composed = rx * rz;
    EXPECT_TRUE(RotateVector(composed, x).isApprox(z, Tolerance<T>));
    EXPECT_TRUE((ToRotationMatrix3(rx) * ToRotationMatrix3(rz))
                    .isApprox(ToRotationMatrix3(composed), Tolerance<T>));
    EXPECT_TRUE(RotateVector(Inverse(rz), y).isApprox(x, Tolerance<T>));
    EXPECT_TRUE(Conjugate(rz).coeffs().isApprox(Inverse(rz).coeffs(), Tolerance<T>));

    const Matrix3<T> rotation = ToRotationMatrix3(composed);
    const Q recovered = QuaternionFromRotationMatrix(rotation);
    EXPECT_TRUE(ToRotationMatrix3(recovered).isApprox(rotation, Tolerance<T>));
    const Matrix4<T> embedded = ToRotationMatrix4(recovered);
    EXPECT_TRUE((embedded.template topLeftCorner<3, 3>().isApprox(rotation, Tolerance<T>)));
    EXPECT_TRUE(TransformPoint(embedded, x).isApprox(RotateVector(recovered, x), Tolerance<T>));

    const Q half = Slerp(identity, rz, T{0.5});
    const V diagonal{std::sqrt(T{0.5}), std::sqrt(T{0.5}), T{0}};
    EXPECT_TRUE(RotateVector(half, x).isApprox(diagonal, Tolerance<T>));
    EXPECT_TRUE(RotateVector(Nlerp(identity, rz, T{0.5}), x).isApprox(diagonal, Tolerance<T>));
    Q negated = rz;
    negated.coeffs() = -negated.coeffs();
    for (T t : {T{0}, T{0.25}, T{0.5}, T{1}})
    {
        EXPECT_TRUE(
            ToRotationMatrix3(Slerp(rz, negated, t)).isApprox(ToRotationMatrix3(rz), Tolerance<T>));
        EXPECT_TRUE(
            ToRotationMatrix3(Nlerp(rz, negated, t)).isApprox(ToRotationMatrix3(rz), Tolerance<T>));
        EXPECT_NEAR(Slerp(identity, rz, t).norm(), T{1}, Tolerance<T>);
    }

    EXPECT_TRUE(RotateVector(RotationBetween(x, y), x).isApprox(y, Tolerance<T>));
    EXPECT_TRUE(RotateVector(RotationBetween(x, V(-x)), x).isApprox(-x, Tolerance<T>));
    EXPECT_TRUE(RotateVector(RotationBetween(x, x), y).isApprox(y, Tolerance<T>));
    const V almostOpposite{T{-1}, T{0.0001}, T{0}};
    EXPECT_TRUE(RotateVector(RotationBetween(x, almostOpposite), x)
                    .isApprox(almostOpposite.normalized(), Tolerance<T>));

    for (T scale : {std::numeric_limits<T>::min(), std::numeric_limits<T>::max() / T{4}})
    {
        Q scaled = rz;
        scaled.coeffs() *= scale;
        EXPECT_TRUE(
            ToRotationMatrix3(Normalize(scaled)).isApprox(ToRotationMatrix3(rz), Tolerance<T>));
        EXPECT_TRUE(RotateVector(QuaternionFromAxisAngle(HalfPi<T>, V(z * scale)), x)
                        .isApprox(y, Tolerance<T>));
    }
}

template <typename T>
void TestTransforms()
{
    using namespace rad;
    using V = Vector3<T>;
    using M = Matrix4<T>;
    const V translation{T{5}, T{-2}, T{7}};
    const V scale{T{2}, T{3}, T{-4}};
    const V axis{T{0}, T{0}, T{1}};
    const Quaternion<T> rotation = QuaternionFromAxisAngle(HalfPi<T>, axis);
    const M transform = ComposeTRS(translation, rotation, scale);
    const V point{T{1}, T{2}, T{3}};
    const V expected{T{-1}, T{0}, T{-5}};

    EXPECT_TRUE((TransformPoint(transform, point) - expected).isZero(Tolerance<T>));
    EXPECT_TRUE(
        TransformDirection(transform, point).isApprox(V{T{-6}, T{2}, T{-12}}, Tolerance<T>));
    EXPECT_TRUE(transform.isApprox(TranslationMatrix(translation) * ToRotationMatrix4(rotation) *
                                       ScaleMatrix(scale),
                                   Tolerance<T>));
    const M identity = M::Identity();
    EXPECT_TRUE(
        TranslationMatrix(translation).isApprox(Translate(identity, translation), Tolerance<T>));
    EXPECT_TRUE(ScaleMatrix(scale).isApprox(Scale(identity, scale), Tolerance<T>));
    EXPECT_TRUE(
        ToRotationMatrix4(rotation).isApprox(Rotate(identity, HalfPi<T>, axis), Tolerance<T>));

    const V tangent{T{1}, T{1}, T{0}};
    const V normal{T{1}, T{-1}, T{0}};
    const V transformedNormal = TransformNormal(transform, normal);
    EXPECT_NEAR(transformedNormal.dot(TransformDirection(transform, tangent)), T{0}, Tolerance<T>);
    EXPECT_TRUE(transformedNormal.isApprox(NormalMatrix(transform) * normal, Tolerance<T>));
    EXPECT_NEAR(transformedNormal.norm(), std::sqrt(T{1} / T{4} + T{1} / T{9}), Tolerance<T>);
    auto checkedNormal = TryNormalMatrix(transform);
    ASSERT_TRUE(checkedNormal);
    EXPECT_TRUE(checkedNormal.value().isApprox(NormalMatrix(transform), Tolerance<T>));

    auto inverse = TryInverse(transform);
    ASSERT_TRUE(inverse);
    EXPECT_TRUE((inverse.value() * transform).isApprox(identity, Tolerance<T>));
    EXPECT_TRUE(TransformPoint(inverse.value(), TransformPoint(transform, point))
                    .isApprox(point, Tolerance<T>));

    for (T magnitude : {T{0.00000000000000000001}, T{100000000000000000000.0}})
    {
        const M scaled = identity * magnitude;
        auto scaledInverse = TryInverse(scaled);
        ASSERT_TRUE(scaledInverse);
        EXPECT_TRUE((scaledInverse.value() * scaled).isApprox(identity, Tolerance<T>));
    }
    const V zeroScale{T{0}, T{1}, T{1}};
    const M singular = ScaleMatrix(zeroScale);
    EXPECT_FALSE(TryInverse(singular));
    EXPECT_EQ(TryNormalMatrix(singular).error(),
              std::make_error_code(std::errc::argument_out_of_domain));
    EXPECT_THROW((void)NormalMatrix(singular), std::system_error);
    EXPECT_TRUE(TransformPoint(singular, point).allFinite());
}

template <typename T>
void TestCamera(bool leftHanded = false)
{
    using namespace rad;
    using V = Vector3<T>;
    using M = Matrix4<T>;
    const V eye{T{3}, T{4}, T{5}};
    const V target{T{-1}, T{1}, T{0}};
    const V up{T{0}, T{2}, T{0}};
    const T viewZ = leftHanded ? T{1} : T{-1};
    const M view = leftHanded ? LookAtLH(eye, target, up) : LookAtRH(eye, target, up);
    EXPECT_TRUE(TransformPoint(view, eye).isZero(Tolerance<T>));
    const V viewedTarget = TransformPoint(view, target);
    EXPECT_NEAR(viewedTarget.x(), T{0}, Tolerance<T>);
    EXPECT_NEAR(viewedTarget.y(), T{0}, Tolerance<T>);
    EXPECT_NEAR(viewedTarget.z(), viewZ * (target - eye).norm(), Tolerance<T>);
    const Matrix3<T> basis = view.template topLeftCorner<3, 3>();
    EXPECT_TRUE((basis * basis.transpose()).isApprox(Matrix3<T>::Identity(), Tolerance<T>));
    EXPECT_NEAR(basis.determinant(), T{1}, Tolerance<T>);

    const T zNear = T{1};
    const T zFar = T{100};
    for (ClipDepth depth : {ClipDepth::ZeroToOne, ClipDepth::NegativeOneToOne})
    {
        const T nearDepth = depth == ClipDepth::ZeroToOne ? T{0} : T{-1};
        const M perspective = leftHanded ? PerspectiveLH(HalfPi<T>, T{2}, zNear, zFar, depth)
                                         : PerspectiveRH(HalfPi<T>, T{2}, zNear, zFar, depth);
        const M frustum = leftHanded ? FrustumLH(T{-2}, T{2}, T{-1}, T{1}, zNear, zFar, depth)
                                     : FrustumRH(T{-2}, T{2}, T{-1}, T{1}, zNear, zFar, depth);
        EXPECT_TRUE(perspective.isApprox(frustum, Tolerance<T>));
        EXPECT_NEAR(ProjectToNdc(perspective, V{T{0}, T{0}, viewZ * zNear}).z(), nearDepth,
                    Tolerance<T>);
        EXPECT_NEAR(ProjectToNdc(perspective, V{T{0}, T{0}, viewZ * zFar}).z(), T{1}, Tolerance<T>);
        EXPECT_TRUE(ProjectToNdc(perspective, V{T{2}, T{1}, viewZ * zNear})
                        .isApprox(V{T{1}, T{1}, nearDepth}, Tolerance<T>));
        EXPECT_GT(ProjectToNdc(perspective, V{T{10}, T{0}, viewZ * zNear}).x(), T{1});
        EXPECT_TRUE(ProjectToNdc(perspective, V{T{1}, T{1}, -viewZ * T{2}}).allFinite());

        const M offCenter = leftHanded ? FrustumLH(T{-1}, T{3}, T{-2}, T{4}, zNear, zFar, depth)
                                       : FrustumRH(T{-1}, T{3}, T{-2}, T{4}, zNear, zFar, depth);
        EXPECT_TRUE(ProjectToNdc(offCenter, V{T{-1}, T{-2}, viewZ * zNear})
                        .isApprox(V{T{-1}, T{-1}, nearDepth}, Tolerance<T>));
        EXPECT_TRUE(ProjectToNdc(offCenter, V{T{3}, T{4}, viewZ * zNear})
                        .isApprox(V{T{1}, T{1}, nearDepth}, Tolerance<T>));
        EXPECT_TRUE(ProjectToNdc(offCenter, V{T{3} * zFar, T{4} * zFar, viewZ * zFar})
                        .isApprox(V{T{1}, T{1}, T{1}}, Tolerance<T>));
        const M ortho = leftHanded ? OrthographicLH(T{-1}, T{3}, T{-2}, T{4}, T{0}, zFar, depth)
                                   : OrthographicRH(T{-1}, T{3}, T{-2}, T{4}, T{0}, zFar, depth);
        EXPECT_TRUE(ProjectToNdc(ortho, V{T{-1}, T{-2}, T{0}})
                        .isApprox(V{T{-1}, T{-1}, nearDepth}, Tolerance<T>));
        EXPECT_TRUE(ProjectToNdc(ortho, V{T{3}, T{4}, viewZ * zFar})
                        .isApprox(V{T{1}, T{1}, T{1}}, Tolerance<T>));

        const M worldToClip = perspective * view;
        auto inverse = TryInverse(worldToClip);
        ASSERT_TRUE(inverse);
        const std::array<V, 3> points{target, V{T{1}, T{2}, T{0}}, V{T{-3}, T{0}, T{-2}}};
        for (const V& point : points)
        {
            const V ndc = ProjectToNdc(worldToClip, point);
            EXPECT_TRUE(
                (UnprojectFromNdc(inverse.value(), ndc) - point).isZero(Tolerance<T> * T{20}));
            ASSERT_TRUE(TryProjectToNdc(worldToClip, point));
            ASSERT_TRUE(TryUnprojectFromNdc(inverse.value(), ndc));
            for (T height : {T{600}, T{-600}})
            {
                const T y = height < T{0} ? T{629} : T{29};
                const Viewport<T> viewport{T{13}, y, T{800}, height, T{0.2}, T{0.8}};
                const V window = ProjectToViewport(worldToClip, point, viewport, depth);
                EXPECT_TRUE(
                    (UnprojectFromViewport(inverse.value(), window, viewport, depth) - point)
                        .isZero(Tolerance<T> * T{30}));
                const V corner = ProjectToViewport(ortho, V{T{-1}, T{-2}, T{0}}, viewport, depth);
                EXPECT_NEAR(corner.x(), viewport.x, Tolerance<T>);
                EXPECT_NEAR(corner.y(), y, Tolerance<T> * T{10});
                EXPECT_NEAR(corner.z(), viewport.minDepth, Tolerance<T>);
            }
        }
        for (T magnitude : {T{0.00000000000000000001}, T{100000000000000000000.0}})
        {
            const M scaled = perspective * magnitude;
            const V point{T{2}, T{3}, viewZ * T{5}};
            EXPECT_TRUE(ProjectToNdc(scaled, point)
                            .isApprox(ProjectToNdc(perspective, point), Tolerance<T>));
        }
    }
}

template <typename T>
void TestLHConventions()
{
    using namespace rad;
    using V = Vector3<T>;
    using M = Matrix4<T>;
    const V zero = V::Zero();
    const V x = V::UnitX();
    const V y = V::UnitY();
    const V z = V::UnitZ();
    const M canonicalView = LookAtLH(zero, z, y);
    EXPECT_TRUE(canonicalView.isIdentity());
    EXPECT_TRUE(TransformPoint(canonicalView, x).isApprox(x));

    const V eye{T{3}, T{4}, T{5}};
    const V target{T{-1}, T{1}, T{0}};
    const M rhView = LookAtRH(eye, target, y);
    M expectedView = rhView;
    expectedView.row(0) *= T{-1};
    expectedView.row(2) *= T{-1};
    EXPECT_TRUE(LookAtLH(eye, target, y).isApprox(expectedView, Tolerance<T>));

    M reverseZ = M::Identity();
    reverseZ(2, 2) = T{-1};
    for (ClipDepth depth : {ClipDepth::ZeroToOne, ClipDepth::NegativeOneToOne})
    {
        EXPECT_TRUE(PerspectiveLH(T{1}, T{2}, T{0.1}, T{100}, depth)
                        .isApprox(PerspectiveRH(T{1}, T{2}, T{0.1}, T{100}, depth) * reverseZ,
                                  Tolerance<T>));
        EXPECT_TRUE(
            FrustumLH(T{-1}, T{3}, T{-2}, T{4}, T{1}, T{100}, depth)
                .isApprox(FrustumRH(T{-1}, T{3}, T{-2}, T{4}, T{1}, T{100}, depth) * reverseZ,
                          Tolerance<T>));
        EXPECT_TRUE(
            OrthographicLH(T{-1}, T{3}, T{-2}, T{4}, T{1}, T{100}, depth)
                .isApprox(OrthographicRH(T{-1}, T{3}, T{-2}, T{4}, T{1}, T{100}, depth) * reverseZ,
                          Tolerance<T>));
    }

    const T nan = std::numeric_limits<T>::quiet_NaN();
    const T infinity = std::numeric_limits<T>::infinity();
    const ClipDepth depth = ClipDepth::ZeroToOne;
    const ClipDepth badDepth = static_cast<ClipDepth>(99);
    const V nonfinite{nan, T{0}, T{0}};
    const V nearlyParallel{T{0}, std::numeric_limits<T>::epsilon(), T{1}};
    EXPECT_THROW((void)LookAtLH(zero, zero, y), std::system_error);
    EXPECT_THROW((void)LookAtLH(zero, z, z), std::system_error);
    EXPECT_THROW((void)LookAtLH(zero, z, nearlyParallel), std::system_error);
    EXPECT_THROW((void)LookAtLH(zero, z, zero), std::system_error);
    EXPECT_THROW((void)LookAtLH(nonfinite, z, y), std::system_error);
    EXPECT_THROW((void)PerspectiveLH(T{0}, T{1}, T{1}, T{10}, depth), std::system_error);
    EXPECT_THROW((void)PerspectiveLH(Pi<T>, T{1}, T{1}, T{10}, depth), std::system_error);
    EXPECT_THROW((void)PerspectiveLH(T{1}, T{0}, T{1}, T{10}, depth), std::system_error);
    EXPECT_THROW((void)PerspectiveLH(T{1}, T{1}, T{0}, T{10}, depth), std::system_error);
    EXPECT_THROW((void)PerspectiveLH(T{1}, T{1}, T{1}, T{1}, depth), std::system_error);
    EXPECT_THROW((void)PerspectiveLH(T{1}, T{1}, T{1}, infinity, depth), std::system_error);
    EXPECT_THROW((void)PerspectiveLH(T{1}, T{1}, T{1}, T{10}, badDepth), std::system_error);
    EXPECT_THROW((void)FrustumLH(T{1}, T{-1}, T{-1}, T{1}, T{1}, T{10}, depth), std::system_error);
    EXPECT_THROW((void)FrustumLH(T{-1}, T{1}, T{-1}, T{1}, T{0}, T{10}, depth), std::system_error);
    EXPECT_THROW((void)FrustumLH(T{-1}, T{1}, T{-1}, T{1}, T{1}, T{10}, badDepth),
                 std::system_error);
    EXPECT_THROW((void)OrthographicLH(T{-1}, T{1}, T{1}, T{1}, T{0}, T{10}, depth),
                 std::system_error);
    EXPECT_THROW((void)OrthographicLH(T{-1}, T{1}, T{-1}, T{1}, T{-1}, T{10}, depth),
                 std::system_error);
    EXPECT_THROW((void)OrthographicLH(T{-1}, T{1}, T{-1}, T{1}, T{0}, T{10}, badDepth),
                 std::system_error);
    EXPECT_THROW((void)OrthographicLH(nan, T{1}, T{-1}, T{1}, T{0}, T{10}, depth),
                 std::system_error);
}

template <typename T>
void TestInvalidInputs()
{
    using namespace rad;
    using V = Vector3<T>;
    using M = Matrix4<T>;
    using Q = Quaternion<T>;
    const V zero = V::Zero();
    const V up{T{0}, T{1}, T{0}};
    const V forward{T{0}, T{0}, T{-1}};
    const M identity = M::Identity();
    const T nan = std::numeric_limits<T>::quiet_NaN();
    const T infinity = std::numeric_limits<T>::infinity();
    const T max = std::numeric_limits<T>::max();
    const ClipDepth depth = ClipDepth::ZeroToOne;
    const ClipDepth badDepth = static_cast<ClipDepth>(99);
    const V nonfinite{nan, T{0}, T{0}};
    const Q invalidQuaternion{T{0}, T{0}, T{0}, T{0}};
    const Q nonunit{T{2}, T{0}, T{0}, T{0}};

    EXPECT_THROW((void)QuaternionFromAxisAngle(T{0}, zero), std::system_error);
    EXPECT_THROW((void)QuaternionFromAxisAngle(nan, up), std::system_error);
    EXPECT_THROW((void)QuaternionFromAxisAngle(T{1}, nonfinite), std::system_error);
    EXPECT_THROW((void)RotationBetween(zero, up), std::system_error);
    EXPECT_THROW((void)Normalize(invalidQuaternion), std::system_error);
    EXPECT_THROW((void)Inverse(nonunit), std::system_error);
    EXPECT_THROW((void)ToRotationMatrix3(nonunit), std::system_error);
    EXPECT_THROW((void)RotateVector(nonunit, up), std::system_error);
    EXPECT_THROW((void)Slerp(Q::Identity(), Q::Identity(), T{-1}), std::system_error);
    EXPECT_THROW((void)Nlerp(Q::Identity(), Q::Identity(), nan), std::system_error);
    const Matrix3<T> reflected = -Matrix3<T>::Identity();
    const Matrix3<T> scaledRotation = Matrix3<T>::Identity() * T{2};
    EXPECT_THROW((void)QuaternionFromRotationMatrix(reflected), std::system_error);
    EXPECT_THROW((void)QuaternionFromRotationMatrix(scaledRotation), std::system_error);

    EXPECT_THROW((void)TranslationMatrix(nonfinite), std::system_error);
    EXPECT_THROW((void)ScaleMatrix(nonfinite), std::system_error);
    EXPECT_THROW((void)ComposeTRS(zero, nonunit, up), std::system_error);
    EXPECT_THROW((void)TransformPoint(identity, nonfinite), std::system_error);
    EXPECT_THROW((void)TransformDirection(identity, nonfinite), std::system_error);
    EXPECT_THROW((void)TransformNormal(identity, nonfinite), std::system_error);
    EXPECT_THROW((void)LookAtRH(zero, zero, up), std::system_error);
    EXPECT_THROW((void)LookAtRH(zero, forward, forward), std::system_error);
    EXPECT_THROW((void)LookAtRH(zero, forward, zero), std::system_error);
    EXPECT_THROW((void)LookAtRH(nonfinite, forward, up), std::system_error);

    EXPECT_THROW((void)PerspectiveRH(T{0}, T{1}, T{1}, T{10}, depth), std::system_error);
    EXPECT_THROW((void)PerspectiveRH(Pi<T>, T{1}, T{1}, T{10}, depth), std::system_error);
    EXPECT_THROW((void)PerspectiveRH(T{1}, T{0}, T{1}, T{10}, depth), std::system_error);
    EXPECT_THROW((void)PerspectiveRH(T{1}, T{1}, T{0}, T{10}, depth), std::system_error);
    EXPECT_THROW((void)PerspectiveRH(T{1}, T{1}, T{1}, T{1}, depth), std::system_error);
    EXPECT_THROW((void)PerspectiveRH(T{1}, T{1}, T{1}, infinity, depth), std::system_error);
    EXPECT_THROW((void)PerspectiveRH(T{1}, T{1}, T{1}, T{10}, badDepth), std::system_error);
    EXPECT_THROW((void)PerspectiveRH(std::nextafter(Pi<T>, T{0}), max, T{1}, T{10}, depth),
                 std::system_error);
    EXPECT_THROW((void)FrustumRH(T{1}, T{-1}, T{-1}, T{1}, T{1}, T{10}, depth), std::system_error);
    EXPECT_THROW((void)OrthographicRH(T{-1}, T{1}, T{1}, T{1}, T{0}, T{10}, depth),
                 std::system_error);
    EXPECT_THROW((void)OrthographicRH(T{-1}, T{1}, T{-1}, T{1}, T{-1}, T{10}, depth),
                 std::system_error);

    const M perspective = PerspectiveRH(T{1}, T{1}, T{1}, T{10}, depth);
    EXPECT_THROW((void)TransformPoint(perspective, up), std::system_error);
    EXPECT_FALSE(TryNormalMatrix(perspective));
    EXPECT_THROW((void)ProjectToNdc(perspective, zero), std::system_error);
    EXPECT_EQ(TryProjectToNdc(perspective, zero).error(),
              std::make_error_code(std::errc::argument_out_of_domain));
    EXPECT_EQ(TryProjectToNdc(identity, nonfinite).error(),
              std::make_error_code(std::errc::invalid_argument));
    const M huge = identity * max;
    EXPECT_EQ(TryProjectToNdc(huge, V{T{2}, T{0}, T{0}}).error(),
              std::make_error_code(std::errc::result_out_of_range));
    const M zeroMatrix = M::Zero();
    EXPECT_FALSE(TryUnprojectFromNdc(zeroMatrix, zero));
    EXPECT_THROW((void)UnprojectFromNdc(zeroMatrix, zero), std::system_error);
    M nonfiniteMatrix = identity;
    nonfiniteMatrix(0, 0) = infinity;
    EXPECT_EQ(TryInverse(nonfiniteMatrix).error(),
              std::make_error_code(std::errc::invalid_argument));

    Viewport<T> viewport;
    viewport.width = T{0};
    EXPECT_THROW((void)ProjectToViewport(identity, zero, viewport, depth), std::system_error);
    EXPECT_THROW((void)UnprojectFromViewport(identity, zero, viewport, depth), std::system_error);
    viewport.width = T{1};
    viewport.maxDepth = viewport.minDepth;
    EXPECT_THROW((void)ProjectToViewport(identity, zero, viewport, depth), std::system_error);
    viewport.maxDepth = T{1};
    EXPECT_THROW((void)UnprojectFromViewport(identity, zero, viewport, badDepth),
                 std::system_error);
    for (T height : {T{0}, -T{0}, nan, infinity, -infinity})
    {
        viewport.height = height;
        EXPECT_THROW((void)ProjectToViewport(identity, zero, viewport, depth), std::system_error);
        EXPECT_THROW((void)UnprojectFromViewport(identity, zero, viewport, depth),
                     std::system_error);
    }
}

template <typename T>
void TestViewport()
{
    using namespace rad;
    using V = Vector3<T>;
    using M = Matrix4<T>;
    const M identity = M::Identity();
    const Viewport<T> defaults;
    EXPECT_EQ(defaults.x, T{0});
    EXPECT_EQ(defaults.y, T{0});
    EXPECT_EQ(defaults.width, T{1});
    EXPECT_EQ(defaults.height, T{1});
    EXPECT_EQ(defaults.minDepth, T{0});
    EXPECT_EQ(defaults.maxDepth, T{1});

    for (ClipDepth depth : {ClipDepth::ZeroToOne, ClipDepth::NegativeOneToOne})
    {
        const T nearDepth = depth == ClipDepth::ZeroToOne ? T{0} : T{-1};
        const T middleDepth = depth == ClipDepth::ZeroToOne ? T{0.5} : T{0};
        const V nearCorner{T{-1}, T{-1}, nearDepth};
        const V farCorner{T{1}, T{1}, T{1}};
        const V center{T{0}, T{0}, middleDepth};
        const V outside{T{2}, T{-2}, T{2}};
        const std::array<Viewport<T>, 2> viewports{
            Viewport<T>{T{13}, T{29}, T{800}, T{600}, T{0.2}, T{0.8}},
            Viewport<T>{T{13}, T{629}, T{800}, T{-600}, T{0.2}, T{0.8}}};
        for (const auto& viewport : viewports)
        {
            const V nearWindow = ProjectToViewport(identity, nearCorner, viewport, depth);
            const V farWindow = ProjectToViewport(identity, farCorner, viewport, depth);
            const V centerWindow = ProjectToViewport(identity, center, viewport, depth);
            EXPECT_NEAR(nearWindow.x(), T{13}, Tolerance<T>);
            EXPECT_NEAR(nearWindow.y(), viewport.height > T{0} ? T{29} : T{629}, Tolerance<T>);
            EXPECT_NEAR(nearWindow.z(), T{0.2}, Tolerance<T>);
            EXPECT_NEAR(farWindow.x(), T{813}, Tolerance<T>);
            EXPECT_NEAR(farWindow.y(), viewport.height > T{0} ? T{629} : T{29}, Tolerance<T>);
            EXPECT_NEAR(farWindow.z(), T{0.8}, Tolerance<T>);
            EXPECT_NEAR(centerWindow.x(), T{413}, Tolerance<T>);
            EXPECT_NEAR(centerWindow.y(), T{329}, Tolerance<T>);
            EXPECT_NEAR(centerWindow.z(), T{0.5}, Tolerance<T>);
            EXPECT_TRUE(UnprojectFromViewport(identity, nearWindow, viewport, depth)
                            .isApprox(nearCorner, Tolerance<T>));
            EXPECT_TRUE(UnprojectFromViewport(identity, farWindow, viewport, depth)
                            .isApprox(farCorner, Tolerance<T>));
            EXPECT_TRUE((UnprojectFromViewport(identity, centerWindow, viewport, depth) - center)
                            .isZero(Tolerance<T>));
            const V outsideWindow = ProjectToViewport(identity, outside, viewport, depth);
            EXPECT_NEAR(outsideWindow.x(), T{1213}, Tolerance<T>);
            EXPECT_NEAR(outsideWindow.y(), viewport.height > T{0} ? T{-271} : T{929}, Tolerance<T>);
            EXPECT_GT(outsideWindow.z(), viewport.maxDepth);
            EXPECT_TRUE(UnprojectFromViewport(identity, outsideWindow, viewport, depth)
                            .isApprox(outside, Tolerance<T>));
        }
    }
}

template <typename T>
void TestNumericalContracts()
{
    using namespace rad;
    using V = Vector3<T>;
    using M = Matrix4<T>;
    using Q = Quaternion<T>;
    const T epsilon = std::numeric_limits<T>::epsilon();
    const V zero = V::Zero();
    const V x = V::UnitX();
    const M identity = M::Identity();
    const Q withinTolerance{T{1} + T{8} * epsilon, T{0}, T{0}, T{0}};
    const Q outsideTolerance{T{1} + T{64} * epsilon, T{0}, T{0}, T{0}};
    EXPECT_NO_THROW((void)ToRotationMatrix3(withinTolerance));
    EXPECT_THROW((void)ToRotationMatrix3(outsideTolerance), std::system_error);

    M approximatelyAffine = identity;
    approximatelyAffine(3, 3) += T{16} * epsilon;
    EXPECT_TRUE(TransformPoint(approximatelyAffine, x).isApprox(x));
    approximatelyAffine(3, 3) = T{1} + T{17} * epsilon;
    EXPECT_THROW((void)TransformPoint(approximatelyAffine, x), std::system_error);

    M illConditioned = identity;
    illConditioned(0, 0) = epsilon / T{8};
    EXPECT_EQ(TryInverse(illConditioned).error(),
              std::make_error_code(std::errc::argument_out_of_domain));
    const Matrix3<T> hugeRotation = Matrix3<T>::Constant(std::numeric_limits<T>::max());
    EXPECT_THROW((void)QuaternionFromRotationMatrix(hugeRotation), std::system_error);

    const T tiny = std::numeric_limits<T>::denorm_min();
    const Q tinyIdentity{tiny, T{0}, T{0}, T{0}};
    EXPECT_TRUE(Normalize(tinyIdentity).coeffs().isApprox(Q::Identity().coeffs()));
    const V tinyAxis{tiny, T{0}, T{0}};
    EXPECT_TRUE(ToRotationMatrix3(QuaternionFromAxisAngle(T{1}, tinyAxis))
                    .isApprox(ToRotationMatrix3(QuaternionFromAxisAngle(T{1}, x)), Tolerance<T>));
    for (T magnitude : {T{0.00000000000000000001}, T{-0.00000000000000000001},
                        T{100000000000000000000.0}, T{-100000000000000000000.0}})
    {
        const M scaled = identity * magnitude;
        EXPECT_TRUE(ProjectToNdc(scaled, x).isApprox(x));
        EXPECT_TRUE(UnprojectFromNdc(scaled, zero).isZero());
    }
    const Q nonfinite{std::numeric_limits<T>::infinity(), T{0}, T{0}, T{0}};
    EXPECT_THROW((void)Normalize(nonfinite), std::system_error);
    EXPECT_THROW((void)Conjugate(nonfinite), std::system_error);
    EXPECT_THROW((void)Slerp(Q::Identity(), outsideTolerance, T{0.5}), std::system_error);
}

template <typename T>
void TestMixedExpressions()
{
    using namespace rad;
    using V = Vector3<T>;
    using M = Matrix4<T>;
    const V zero = V::Zero();
    const V x = V::UnitX();
    const V y = V::UnitY();
    const V z = V::UnitZ();
    const M identity = M::Identity();

    EXPECT_TRUE(RotateVector(RotationBetween(x, V::UnitY()), x).isApprox(y, Tolerance<T>));
    EXPECT_TRUE(RotateVector(RotationBetween(V::UnitX(), y), x).isApprox(y, Tolerance<T>));
    EXPECT_TRUE(LookAtLH(V::Zero(), z, y).isIdentity());
    EXPECT_TRUE(LookAtLH(zero, V::UnitZ(), y).isIdentity());
    EXPECT_TRUE(LookAtLH(zero, z, V::UnitY()).isIdentity());
    EXPECT_TRUE(LookAtRH(V::Zero(), V(-z), y).isIdentity());
    EXPECT_TRUE(LookAtRH(zero, -V::UnitZ(), y).isIdentity());
    EXPECT_TRUE(LookAtRH(zero, V(-z), V::UnitY()).isIdentity());

    EXPECT_TRUE(TransformPoint(identity, V::UnitX()).isApprox(x));
    EXPECT_TRUE(TransformPoint(M::Identity(), x).isApprox(x));
    EXPECT_TRUE(TransformDirection(identity, V::UnitX()).isApprox(x));
    EXPECT_TRUE(TransformDirection(M::Identity(), x).isApprox(x));
    EXPECT_TRUE(TransformNormal(identity, V::UnitX()).isApprox(x));
    EXPECT_TRUE(TransformNormal(M::Identity(), x).isApprox(x));
    EXPECT_TRUE(ProjectToNdc(identity, V::UnitX()).isApprox(x));
    EXPECT_TRUE(ProjectToNdc(M::Identity(), x).isApprox(x));
    EXPECT_TRUE(UnprojectFromNdc(identity, V::UnitX()).isApprox(x));
    EXPECT_TRUE(UnprojectFromNdc(M::Identity(), x).isApprox(x));
    ASSERT_TRUE(TryProjectToNdc(identity, V::UnitX()));
    ASSERT_TRUE(TryProjectToNdc(M::Identity(), x));
    ASSERT_TRUE(TryUnprojectFromNdc(identity, V::UnitX()));
    ASSERT_TRUE(TryUnprojectFromNdc(M::Identity(), x));

    const Eigen::MatrixBase<M>& matrixBase = identity;
    const Eigen::MatrixBase<V>& vectorBase = x;
    EXPECT_TRUE(TransformPoint(matrixBase, vectorBase).isApprox(x));
    EXPECT_TRUE(TransformDirection(matrixBase, vectorBase).isApprox(x));
    EXPECT_TRUE(TransformNormal(matrixBase, vectorBase).isApprox(x));
    EXPECT_TRUE(ProjectToNdc(matrixBase, vectorBase).isApprox(x));
    EXPECT_TRUE(UnprojectFromNdc(matrixBase, vectorBase).isApprox(x));
    EXPECT_TRUE(TranslationMatrix(vectorBase).isApprox(TranslationMatrix(x)));
    EXPECT_TRUE(NormalMatrix(matrixBase).isIdentity());
    ASSERT_TRUE(TryNormalMatrix(matrixBase));
    ASSERT_TRUE(TryInverse(matrixBase));
}

} // namespace

TEST(Math, Transform3DQuaternions)
{
    TestQuaternions<float>();
    TestQuaternions<double>();
}

TEST(Math, Transform3DTransforms)
{
    TestTransforms<float>();
    TestTransforms<double>();
}

TEST(Math, Transform3DCamera)
{
    TestCamera<float>();
    TestCamera<double>();
}

TEST(Math, Transform3DCameraLH)
{
    TestCamera<float>(true);
    TestCamera<double>(true);
}

TEST(Math, Transform3DLHConventions)
{
    TestLHConventions<float>();
    TestLHConventions<double>();
}

TEST(Math, Transform3DInvalidInputs)
{
    TestInvalidInputs<float>();
    TestInvalidInputs<double>();
}

TEST(Math, Transform3DNumericalContracts)
{
    TestNumericalContracts<float>();
    TestNumericalContracts<double>();
}

TEST(Math, Transform3DViewport)
{
    TestViewport<float>();
    TestViewport<double>();
}

TEST(Math, Transform3DExpressions)
{
    using namespace rad;
    const auto translation = TranslationMatrix(Vector3f::Zero());
    const auto scale = ScaleMatrix(Vector3d::Ones());
    const auto point = TransformPoint(Matrix4f::Identity(), Vector3f::Zero());
    const auto direction = TransformDirection(Matrix4d::Identity(), Vector3d::Ones());
    const auto normalMatrix = NormalMatrix(Matrix4f::Identity());
    const auto normal = TransformNormal(Matrix4d::Identity(), Vector3d::UnitY());
    const auto inverse = TryInverse(Matrix4f::Identity());
    const auto checkedNormal = TryNormalMatrix(Matrix4d::Identity());
    const auto rotation = QuaternionFromRotationMatrix(Matrix3f::Identity());
    const auto between = RotationBetween(Vector3d::UnitX(), Vector3d::UnitY());
    const auto view = LookAtRH(Vector3f::Zero(), -Vector3f::UnitZ(), Vector3f::UnitY());
    const auto lhView = LookAtLH(Vector3d::Zero(), Vector3d::UnitZ(), Vector3d::UnitY());
    const auto ndc = ProjectToNdc(Matrix4d::Identity(), Vector3d::Ones() * 2.0);
    const auto world = UnprojectFromNdc(Matrix4f::Identity(), Vector3f::Ones() * 2.0f);
    const auto checkedNdc = TryProjectToNdc(Matrix4f::Identity(), Vector3f::Ones());
    const auto checkedWorld = TryUnprojectFromNdc(Matrix4d::Identity(), Vector3d::Ones());
    EXPECT_TRUE(translation.isIdentity());
    EXPECT_TRUE(scale.isIdentity());
    EXPECT_TRUE(point.isZero());
    EXPECT_TRUE(direction.isApprox(Vector3d::Ones()));
    EXPECT_TRUE(normalMatrix.isIdentity());
    EXPECT_TRUE(normal.isApprox(Vector3d::UnitY()));
    ASSERT_TRUE(inverse);
    ASSERT_TRUE(checkedNormal);
    EXPECT_TRUE(ToRotationMatrix3(rotation).isIdentity());
    EXPECT_TRUE(RotateVector(between, Vector3d::UnitX()).isApprox(Vector3d::UnitY()));
    EXPECT_TRUE(view.isIdentity());
    EXPECT_TRUE(lhView.isIdentity());
    EXPECT_TRUE(ndc.isApprox(Vector3d::Ones() * 2.0));
    EXPECT_TRUE(world.isApprox(Vector3f::Ones() * 2.0f));
    ASSERT_TRUE(checkedNdc);
    ASSERT_TRUE(checkedWorld);
}

TEST(Math, Transform3DMixedExpressions)
{
    TestMixedExpressions<float>();
    TestMixedExpressions<double>();
}
