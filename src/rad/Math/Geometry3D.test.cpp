#include <rad/Math/Geometry3D.h>

#include <gtest/gtest.h>

TEST(Math, Geometry3DCompatibility)
{
    const rad::Vector3f x = rad::Vector3f::UnitX();
    const rad::Vector3f y = rad::Vector3f::UnitY();
    const rad::Quaternionf rotation = rad::QuaternionFromAxisAngle(rad::HalfPi<float>, y);
    const rad::Matrix4f transform = rad::ToRotationMatrix4(rotation);
    EXPECT_TRUE(rad::TransformPoint(transform, x).isApprox(-rad::Vector3f::UnitZ()));
    EXPECT_TRUE(rad::TryInverse(transform).has_value());
    EXPECT_TRUE(rad::LookAtLH(rad::Vector3f::Zero(), rad::Vector3f::UnitZ(), y).isIdentity());
    const auto projection =
        rad::PerspectiveRH(rad::HalfPi<float>, 1.0f, 1.0f, 10.0f, rad::ClipDepth::ZeroToOne);
    const rad::Vector3f nearPoint{0.0f, 0.0f, -1.0f};
    EXPECT_TRUE(rad::ProjectToNdc(projection, nearPoint).isZero());
}
