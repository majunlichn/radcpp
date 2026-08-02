#include <rad/Core/Float8.h>

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <type_traits>

static_assert(sizeof(rad::Float8E4M3) == 1);
static_assert(sizeof(rad::Float8E5M2) == 1);
static_assert(std::is_trivially_copyable_v<rad::Float8E4M3>);
static_assert(std::is_trivially_copyable_v<rad::Float8E5M2>);
static_assert(rad::fp8e4m3fn_to_fp32_value(0x38) == 1.0f);
static_assert(rad::fp8e4m3fn_from_fp32_value(1.0f) == 0x38);
static_assert(rad::fp8e5m2_to_fp32_value(0x3C) == 1.0f);
static_assert(rad::fp8e5m2_from_fp32_value(1.0f) == 0x3C);
static_assert(rad::Float8E4M3{1.0f}.Bits() == 0x38);
static_assert(rad::Float8E5M2{1.0f}.Bits() == 0x3C);
static_assert(std::numeric_limits<rad::Float8E4M3>::max().Bits() == 0x7E);
static_assert(std::numeric_limits<rad::Float8E5M2>::max().Bits() == 0x7B);
static_assert(std::numeric_limits<rad::Float8E4M3>::max_exponent == 9);

TEST(Core, Float8E4M3Conversion)
{
    EXPECT_FLOAT_EQ(rad::fp8e4m3fn_to_fp32_value(0x00), 0.0f);
    EXPECT_FLOAT_EQ(rad::fp8e4m3fn_to_fp32_value(0x80), -0.0f);
    EXPECT_FLOAT_EQ(rad::fp8e4m3fn_to_fp32_value(0x01), 0x1p-9f);
    EXPECT_FLOAT_EQ(rad::fp8e4m3fn_to_fp32_value(0x08), 0x1p-6f);
    EXPECT_FLOAT_EQ(rad::fp8e4m3fn_to_fp32_value(0x7E), 448.0f);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(rad::fp8e4m3fn_to_fp32_value(0x7F)), 0x7FF00000u);
    EXPECT_TRUE(std::isnan(rad::fp8e4m3fn_to_fp32_value(0x7F)));

    EXPECT_EQ(rad::fp8e4m3fn_from_fp32_value(448.0f), 0x7E);
    EXPECT_EQ(rad::fp8e4m3fn_from_fp32_value(480.0f), 0x7E);
    EXPECT_EQ(rad::fp8e4m3fn_from_fp32_value(std::numeric_limits<float>::infinity()), 0x7E);
    EXPECT_EQ(rad::fp8e4m3fn_from_fp32_value(std::numeric_limits<float>::quiet_NaN()), 0x7F);
}

TEST(Core, Float8E5M2Conversion)
{
    EXPECT_FLOAT_EQ(rad::fp8e5m2_to_fp32_value(0x00), 0.0f);
    EXPECT_FLOAT_EQ(rad::fp8e5m2_to_fp32_value(0x80), -0.0f);
    EXPECT_FLOAT_EQ(rad::fp8e5m2_to_fp32_value(0x01), 0x1p-16f);
    EXPECT_FLOAT_EQ(rad::fp8e5m2_to_fp32_value(0x04), 0x1p-14f);
    EXPECT_FLOAT_EQ(rad::fp8e5m2_to_fp32_value(0x7B), 57344.0f);
    EXPECT_TRUE(std::isinf(rad::fp8e5m2_to_fp32_value(0x7C)));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(rad::fp8e5m2_to_fp32_value(0x7D)), 0x7FA00000u);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(rad::fp8e5m2_to_fp32_value(0x7E)), 0x7FC00000u);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(rad::fp8e5m2_to_fp32_value(0x7F)), 0x7FE00000u);
    EXPECT_TRUE(std::isnan(rad::fp8e5m2_to_fp32_value(0x7F)));

    EXPECT_EQ(rad::fp8e5m2_from_fp32_value(57344.0f), 0x7B);
    EXPECT_EQ(rad::fp8e5m2_from_fp32_value(65536.0f), 0x7C);
    EXPECT_EQ(rad::fp8e5m2_from_fp32_value(std::numeric_limits<float>::infinity()), 0x7C);
    EXPECT_EQ(rad::fp8e5m2_from_fp32_value(std::numeric_limits<float>::quiet_NaN()), 0x7F);
}

TEST(Core, Float8ConversionRoundTrip)
{
    for (std::uint32_t rawBits = 0; rawBits <= std::numeric_limits<std::uint8_t>::max(); ++rawBits)
    {
        SCOPED_TRACE(rawBits);
        const auto bits = static_cast<std::uint8_t>(rawBits);

        const std::uint8_t expectedE4M3 = bits;
        EXPECT_EQ(rad::fp8e4m3fn_from_fp32_value(rad::fp8e4m3fn_to_fp32_value(bits)), expectedE4M3);

        const bool isE5M2NaN = (bits & 0x7Cu) == 0x7Cu && (bits & 0x03u) != 0;
        const std::uint8_t expectedE5M2 =
            isE5M2NaN ? static_cast<std::uint8_t>((bits & 0x80u) | 0x7Fu) : bits;
        EXPECT_EQ(rad::fp8e5m2_from_fp32_value(rad::fp8e5m2_to_fp32_value(bits)), expectedE5M2);
    }
}

TEST(Core, Float8RoundsToNearestEven)
{
    EXPECT_EQ(rad::fp8e4m3fn_from_fp32_value(1.0625f), 0x38);
    EXPECT_EQ(rad::fp8e4m3fn_from_fp32_value(1.1875f), 0x3A);
    EXPECT_EQ(rad::fp8e4m3fn_from_fp32_value(-1.0625f), 0xB8);
    EXPECT_EQ(rad::fp8e4m3fn_from_fp32_value(-1.1875f), 0xBA);

    EXPECT_EQ(rad::fp8e5m2_from_fp32_value(1.125f), 0x3C);
    EXPECT_EQ(rad::fp8e5m2_from_fp32_value(1.375f), 0x3E);
    EXPECT_EQ(rad::fp8e5m2_from_fp32_value(-1.125f), 0xBC);
    EXPECT_EQ(rad::fp8e5m2_from_fp32_value(-1.375f), 0xBE);
}

TEST(Core, Float8ArithmeticAndClassification)
{
    {
        rad::Float8E4M3 a = 2.0f;
        const rad::Float8E4M3 b = 4.0f;
        EXPECT_FLOAT_EQ(static_cast<float>(a + b), 6.0f);
        a *= b;
        EXPECT_FLOAT_EQ(static_cast<float>(a), 8.0f);
        EXPECT_DOUBLE_EQ(a + 0.5, 8.5);
        EXPECT_FLOAT_EQ(static_cast<float>(a + 2), 10.0f);
        EXPECT_TRUE(rad::Float8E4M3::FromBits(0x7F).IsNaN());
        EXPECT_FALSE(a.IsInfinite());
    }

    {
        const rad::Float8E5M2 a = 2.0f;
        const rad::Float8E5M2 b = 4.0f;
        EXPECT_FLOAT_EQ(static_cast<float>(a * b), 8.0f);
        EXPECT_TRUE(a < b);
        EXPECT_TRUE(rad::Float8E5M2::FromBits(0x7C).IsInfinite());
        EXPECT_TRUE(rad::Float8E5M2::FromBits(0x7F).IsNaN());
    }
}

TEST(Core, Float8Streaming)
{
    std::ostringstream stream;
    stream << rad::Float8E4M3{1.5f} << ' ' << rad::Float8E5M2{1.5f};
    EXPECT_EQ(stream.str(), "1.5 1.5");
}
