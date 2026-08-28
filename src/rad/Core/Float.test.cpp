#include <rad/Core/Float.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

static_assert(rad::FloatingPoint<float>);
static_assert(rad::FloatingPoint<const double>);
static_assert(!rad::FloatingPoint<int>);

static_assert(rad::FloatEpsilon<float> == std::numeric_limits<float>::epsilon());
static_assert(rad::Float32ToBits(1.0f) == 0x3F800000u);
static_assert(rad::Float32FromBits(0x3F800000u) == 1.0f);
static_assert(rad::Float64ToBits(1.0) == 0x3FF0000000000000ull);
static_assert(rad::Float64FromBits(0x3FF0000000000000ull) == 1.0);

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
