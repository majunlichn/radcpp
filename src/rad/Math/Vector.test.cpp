#include <rad/Math/Vector.h>

#include <gtest/gtest.h>

static_assert(requires(const rad::Vector3f& value) {
    rad::DegreesToRadians(value);
    rad::RadiansToDegrees(value);
});

TEST(Math, VectorAngleConversions)
{
    const rad::Vector3d degrees{0.0, 90.0, 180.0};
    const rad::Vector3d radians = rad::DegreesToRadians(degrees);
    const auto radians1 = rad::DegreesToRadians(rad::Vector3d{0.0, 90.0, 180.0});

    EXPECT_TRUE(radians.isApprox(rad::Vector3d{0.0, rad::HalfPi<double>, rad::Pi<double>}));
    EXPECT_TRUE(radians1.isApprox(radians));
    EXPECT_TRUE(rad::RadiansToDegrees(radians).isApprox(degrees));
}
