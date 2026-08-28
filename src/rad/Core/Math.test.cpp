#include <rad/Core/Math.h>

#include <gtest/gtest.h>

#include <cstdint>

static_assert(rad::AlmostEqual(rad::Pi<double>, 3.14159265358979323846));
static_assert(rad::AlmostEqual(rad::TwoPi<double>, rad::Pi<double> * 2.0));
static_assert(rad::AlmostEqual(rad::HalfPi<double>, rad::Pi<double> / 2.0));
static_assert(rad::Abs(-2.5f) == 2.5f);
static_assert(rad::Clamp(1.5f, 0.0f, 1.0f) == 1.0f);
static_assert(rad::Lerp(10.0f, 20.0f, 0.25f) == 12.5f);
static_assert(rad::QuantizeUnorm<std::uint8_t>(0.5f, 0.0f, 1.0f) == 128);
static_assert(rad::QuantizeSnorm<std::int8_t>(-0.5f, -1.0f, 1.0f) == -64);

TEST(Core, MathAngles)
{
    EXPECT_TRUE(rad::AlmostEqual(rad::Degrees(rad::Radians(90.0)), 90.0));
}

TEST(Core, MathExponentialFunctions)
{
    EXPECT_DOUBLE_EQ(rad::Rsqrt(4.0), 0.5);
}

TEST(Core, MathCommonFunctions)
{
    EXPECT_DOUBLE_EQ(rad::Sign(-2.0), -1.0);
    EXPECT_DOUBLE_EQ(rad::RoundEven(2.5), 2.0);
    EXPECT_DOUBLE_EQ(rad::Fract(-1.5), 0.5);
    EXPECT_DOUBLE_EQ(rad::Mod(-1.5, 2.0), 0.5);
    EXPECT_DOUBLE_EQ(rad::Mix(2.0, 4.0, 0.25), 2.5);
    EXPECT_DOUBLE_EQ(rad::Step(0.0, -1.0), 0.0);

    double integralPart;
    EXPECT_DOUBLE_EQ(rad::Modf(-1.5, integralPart), -0.5);
    EXPECT_DOUBLE_EQ(integralPart, -1.0);

    int exponent;
    EXPECT_DOUBLE_EQ(rad::Frexp(3.0, exponent), 0.75);
    EXPECT_EQ(exponent, 2);
}

TEST(Core, MathInterpolation)
{
    EXPECT_FLOAT_EQ(rad::InverseLerp(2.0f, 6.0f, 3.0f), 0.25f);
    EXPECT_FLOAT_EQ(rad::Remap(0.0f, 10.0f, -1.0f, 1.0f, 7.5f), 0.5f);
    EXPECT_FLOAT_EQ(rad::SmoothStep(0.0f, 1.0f, 0.25f), 0.15625f);
}

TEST(Core, MathQuantization)
{
    EXPECT_EQ(rad::QuantizeUnorm8(0.5f, 0.0f, 1.0f), 128);
    EXPECT_FLOAT_EQ(rad::DequantizeUnorm16(65535), 1.0f);
    EXPECT_EQ(rad::QuantizeSnorm<std::int8_t>(-2.0f, -1.0f, 1.0f), -127);
    EXPECT_FLOAT_EQ(rad::DequantizeSnorm<std::int8_t>(-128), -1.0f);
}
