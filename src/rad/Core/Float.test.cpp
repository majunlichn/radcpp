#include <rad/Core/Float.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

static_assert(rad::FloatingPoint<float>);
static_assert(rad::FloatingPoint<const double>);
static_assert(!rad::FloatingPoint<int>);

static_assert(rad::AlmostEqual(rad::Pi<double>, 3.14159265358979323846));
static_assert(rad::AlmostEqual(rad::TwoPi<double>, rad::Pi<double> * 2.0));
static_assert(rad::AlmostEqual(rad::HalfPi<double>, rad::Pi<double> / 2.0));
static_assert(rad::FloatEpsilon<float> == std::numeric_limits<float>::epsilon());
static_assert(rad::Float32ToBits(1.0f) == 0x3F800000u);
static_assert(rad::Float32FromBits(0x3F800000u) == 1.0f);
static_assert(rad::Float64ToBits(1.0) == 0x3FF0000000000000ull);
static_assert(rad::Float64FromBits(0x3FF0000000000000ull) == 1.0);
static_assert(rad::AlmostEqual(rad::DegreesToRadians(180.0), rad::Pi<double>));
static_assert(rad::AlmostEqual(rad::RadiansToDegrees(rad::Pi<double>), 180.0));
static_assert(rad::Abs(-2.5f) == 2.5f);
static_assert(rad::Clamp(1.5f, 0.0f, 1.0f) == 1.0f);
static_assert(rad::Saturate(-0.25f) == 0.0f);
static_assert(rad::Lerp(10.0f, 20.0f, 0.25f) == 12.5f);
static_assert(rad::InverseLerp(10.0f, 20.0f, 12.5f) == 0.25f);
static_assert(rad::Remap(0.0f, 10.0f, 100.0f, 200.0f, 2.5f) == 125.0f);
static_assert(rad::Normalize(5.0f, 0.0f, 10.0f) == 0.5f);
static_assert(rad::Normalize(-5.0f, 0.0f, 10.0f) == 0.0f);
static_assert(rad::Normalize(15.0f, 0.0f, 10.0f) == 1.0f);
static_assert(rad::SmoothStep(0.0f, 1.0f, -1.0f) == 0.0f);
static_assert(rad::SmoothStep(0.0f, 1.0f, 0.5f) == 0.5f);
static_assert(rad::SmoothStep(0.0f, 1.0f, 2.0f) == 1.0f);
static_assert(rad::QuantizeUnorm<std::uint8_t>(0.0f, 0.0f, 1.0f) == 0);
static_assert(rad::QuantizeUnorm<std::uint8_t>(0.5f, 0.0f, 1.0f) == 128);
static_assert(rad::QuantizeUnorm<std::uint8_t>(1.0f, 0.0f, 1.0f) == 255);
static_assert(rad::DequantizeUnorm<std::uint8_t>(255) == 1.0f);

TEST(Core, FloatComparisons)
{
    EXPECT_TRUE(rad::AlmostZero(1.0e-7f, 1.0e-6f));
    EXPECT_FALSE(rad::AlmostZero(1.0e-5f, 1.0e-6f));
    EXPECT_FALSE(rad::AlmostZero(std::numeric_limits<float>::infinity()));
    EXPECT_FALSE(rad::AlmostZero(std::numeric_limits<float>::quiet_NaN()));

    EXPECT_TRUE(rad::AlmostEqual(0.1 + 0.2, 0.3));
    EXPECT_TRUE(rad::AlmostEqual(0.0, std::numeric_limits<double>::epsilon()));
    EXPECT_FALSE(rad::AlmostEqual(0.0, std::numeric_limits<double>::epsilon() * 2.0));
    EXPECT_TRUE(rad::AlmostEqual(1'000'000.0, 1'000'000.25, 1.0e-6));
    EXPECT_FALSE(rad::AlmostEqual(1.0, 1.1, 1.0e-6));

    EXPECT_TRUE(rad::AlmostEqual(std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::infinity()));
    EXPECT_FALSE(rad::AlmostEqual(std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::max()));
    EXPECT_FALSE(rad::AlmostEqual(std::numeric_limits<double>::quiet_NaN(), 0.0));
}

TEST(Core, FloatInterpolation)
{
    EXPECT_FLOAT_EQ(rad::Lerp(2.0f, 6.0f, 0.0f), 2.0f);
    EXPECT_FLOAT_EQ(rad::Lerp(2.0f, 6.0f, 0.5f), 4.0f);
    EXPECT_FLOAT_EQ(rad::Lerp(2.0f, 6.0f, 1.0f), 6.0f);

    EXPECT_FLOAT_EQ(rad::InverseLerp(2.0f, 6.0f, 3.0f), 0.25f);
    EXPECT_FLOAT_EQ(rad::Remap(0.0f, 10.0f, -1.0f, 1.0f, 7.5f), 0.5f);
    EXPECT_FLOAT_EQ(rad::SmoothStep(0.0f, 1.0f, 0.25f), 0.15625f);
}

TEST(Core, FloatNormalization)
{
    const float quietNaN = std::numeric_limits<float>::quiet_NaN();

    EXPECT_FLOAT_EQ(rad::Clamp(-1.0f, 0.0f, 1.0f), 0.0f);
    EXPECT_FLOAT_EQ(rad::Clamp(2.0f, 0.0f, 1.0f), 1.0f);
    EXPECT_TRUE(rad::IsNaN(rad::Clamp(quietNaN, 0.0f, 1.0f)));
    EXPECT_TRUE(rad::IsNaN(rad::Saturate(quietNaN)));

    EXPECT_FLOAT_EQ(rad::Normalize(5.0f, 0.0f, 10.0f), 0.5f);
    EXPECT_FLOAT_EQ(rad::Normalize(-5.0f, 0.0f, 10.0f), 0.0f);
    EXPECT_FLOAT_EQ(rad::Normalize(15.0f, 0.0f, 10.0f), 1.0f);
    EXPECT_TRUE(rad::IsNaN(rad::Normalize(quietNaN, 0.0f, 10.0f)));

    EXPECT_EQ(rad::QuantizeUnorm8(-1.0f, 0.0f, 1.0f), 0);
    EXPECT_EQ(rad::QuantizeUnorm8(0.0f, 0.0f, 1.0f), 0);
    EXPECT_EQ(rad::QuantizeUnorm8(0.5f, 0.0f, 1.0f), 128);
    EXPECT_EQ(rad::QuantizeUnorm8(1.0f, 0.0f, 1.0f), 255);
    EXPECT_EQ(rad::QuantizeUnorm8(2.0f, 0.0f, 1.0f), 255);
    EXPECT_EQ(rad::QuantizeUnorm8(15.0f, 10.0f, 20.0f), 128);
    EXPECT_EQ(rad::QuantizeUnorm16(0.5f, 0.0f, 1.0f), 32768);

    EXPECT_FLOAT_EQ(rad::DequantizeUnorm8(0), 0.0f);
    EXPECT_FLOAT_EQ(rad::DequantizeUnorm8(255), 1.0f);
    EXPECT_FLOAT_EQ(rad::DequantizeUnorm16(65535), 1.0f);
}

TEST(Core, FloatAngles)
{
    EXPECT_TRUE(rad::AlmostEqual(rad::DegreesToRadians(180.0), rad::Pi<double>));
    EXPECT_TRUE(rad::AlmostEqual(rad::RadiansToDegrees(rad::Pi<double>), 180.0));
    EXPECT_TRUE(rad::AlmostEqual(rad::DegreesToRadians(90.0), rad::Pi<double> / 2.0));
    EXPECT_TRUE(rad::AlmostEqual(rad::RadiansToDegrees(rad::Pi<double> / 2.0), 90.0));
}

TEST(Core, FloatBits)
{
    EXPECT_EQ(rad::Float32ToBits(1.0f), 0x3F800000u);
    EXPECT_EQ(rad::Float32FromBits(0x3F800000u), 1.0f);
    EXPECT_EQ(rad::Float64ToBits(1.0), 0x3FF0000000000000ull);
    EXPECT_EQ(rad::Float64FromBits(0x3FF0000000000000ull), 1.0);
}

TEST(Core, FloatClassification)
{
    EXPECT_TRUE(rad::IsFinite(1.0));
    EXPECT_FALSE(rad::IsFinite(std::numeric_limits<double>::infinity()));
    EXPECT_TRUE(rad::IsInfinite(std::numeric_limits<double>::infinity()));
    EXPECT_FALSE(rad::IsInfinite(1.0));
    EXPECT_TRUE(rad::IsNaN(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_FALSE(rad::IsNaN(1.0));
}
