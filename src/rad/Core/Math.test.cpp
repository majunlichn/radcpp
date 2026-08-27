#include <rad/Core/Math.h>

#include <gtest/gtest.h>

static_assert(rad::AlmostEqual(rad::DegreesToRadians(180.0), rad::Pi<double>));
static_assert(rad::AlmostEqual(rad::RadiansToDegrees(rad::Pi<double>), 180.0));

TEST(Core, MathAngles)
{
    EXPECT_TRUE(rad::AlmostEqual(rad::DegreesToRadians(180.0), rad::Pi<double>));
    EXPECT_TRUE(rad::AlmostEqual(rad::RadiansToDegrees(rad::Pi<double>), 180.0));
    EXPECT_TRUE(rad::AlmostEqual(rad::DegreesToRadians(90.0), rad::Pi<double> / 2.0));
    EXPECT_TRUE(rad::AlmostEqual(rad::RadiansToDegrees(rad::Pi<double> / 2.0), 90.0));
}
