#include <rad/Core/BFloat16.h>

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <limits>
#include <sstream>
#include <type_traits>

static_assert(sizeof(rad::BFloat16) == 2);
static_assert(std::is_trivially_copyable_v<rad::BFloat16>);
static_assert(rad::bf16_to_fp32(0x3F80) == 1.0f);
static_assert(rad::bf16_from_fp32_round_to_zero(1.0f) == 0x3F80);
static_assert(rad::bf16_from_fp32_round_to_nearest_even(1.0f) == 0x3F80);
static_assert(rad::BFloat16::FromBits(0x3F80).Bits() == 0x3F80);
static_assert(rad::BFloat16{1.0f}.Bits() == 0x3F80);
static_assert(static_cast<float>(rad::BFloat16::FromBits(0x4000)) == 2.0f);
static_assert((rad::BFloat16{1.0f} + rad::BFloat16{2.0f}).Bits() == 0x4040);
static_assert(std::same_as<decltype(rad::BFloat16{} + 1.0), double>);
static_assert(std::same_as<decltype(rad::BFloat16{} + 1), rad::BFloat16>);
static_assert((rad::BFloat16{2.0f} + 3).Bits() == 0x40A0);
static_assert(std::numeric_limits<rad::BFloat16>::digits == 8);
static_assert(std::numeric_limits<rad::BFloat16>::min().Bits() == 0x0080);
static_assert(std::numeric_limits<rad::BFloat16>::lowest().Bits() == 0xFF7F);
static_assert(std::numeric_limits<rad::BFloat16>::max().Bits() == 0x7F7F);
static_assert(std::numeric_limits<rad::BFloat16>::epsilon().Bits() == 0x3C00);
static_assert(std::numeric_limits<rad::BFloat16>::infinity().Bits() == 0x7F80);
static_assert(std::numeric_limits<rad::BFloat16>::quiet_NaN().Bits() == 0x7FC0);
static_assert(std::numeric_limits<rad::BFloat16>::denorm_min().Bits() == 0x0001);

TEST(Core, BFloat16Conversion)
{
    EXPECT_EQ(rad::BFloat16{}.Bits(), 0x0000);
    EXPECT_EQ(rad::BFloat16{1.0f}.Bits(), 0x3F80);
    EXPECT_EQ(rad::BFloat16{-2.0f}.Bits(), 0xC000);
    EXPECT_EQ(rad::BFloat16{3.14159265f}.Bits(), 0x4049);
    EXPECT_FLOAT_EQ(static_cast<float>(rad::BFloat16::FromBits(0x3EAB)), 0.333984375f);

    EXPECT_EQ(rad::BFloat16{std::numeric_limits<float>::infinity()}.Bits(), 0x7F80);
    EXPECT_EQ(rad::BFloat16{-std::numeric_limits<float>::infinity()}.Bits(), 0xFF80);
    EXPECT_EQ(rad::BFloat16{std::numeric_limits<float>::quiet_NaN()}.Bits(), 0x7FC0);
    EXPECT_EQ(rad::bf16_from_fp32_round_to_zero(std::numeric_limits<float>::quiet_NaN()), 0x7FC0);
    EXPECT_EQ(rad::bf16_from_fp32_round_to_nearest_even(std::numeric_limits<float>::quiet_NaN()),
              0x7FC0);
}

TEST(Core, BFloat16ConversionRoundTrip)
{
    for (std::uint32_t rawBits = 0; rawBits <= std::numeric_limits<std::uint16_t>::max(); ++rawBits)
    {
        SCOPED_TRACE(rawBits);
        const auto bits = static_cast<std::uint16_t>(rawBits);
        const float value = rad::bf16_to_fp32(bits);
        const bool isNaN = (bits & 0x7F80u) == 0x7F80u && (bits & 0x007Fu) != 0;
        const std::uint16_t expectedBits = isNaN ? 0x7FC0u : bits;

        EXPECT_EQ(std::bit_cast<std::uint32_t>(value), rawBits << 16u);
        EXPECT_EQ(rad::bf16_from_fp32_round_to_zero(value), expectedBits);
        EXPECT_EQ(rad::bf16_from_fp32_round_to_nearest_even(value), expectedBits);
    }
}

TEST(Core, BFloat16RoundsToNearestEven)
{
    const float evenTie = std::bit_cast<float>(std::uint32_t{0x3F808000});
    const float oddTie = std::bit_cast<float>(std::uint32_t{0x3F818000});
    const float roundsUp = std::bit_cast<float>(std::uint32_t{0x3F80FFFF});
    const float negativeEvenTie = std::bit_cast<float>(std::uint32_t{0xBF808000});
    const float negativeOddTie = std::bit_cast<float>(std::uint32_t{0xBF818000});
    const float negativeRoundsUp = std::bit_cast<float>(std::uint32_t{0xBF80FFFF});

    EXPECT_EQ(rad::bf16_from_fp32_round_to_zero(roundsUp), 0x3F80);
    EXPECT_EQ(rad::bf16_from_fp32_round_to_nearest_even(roundsUp), 0x3F81);
    EXPECT_EQ(rad::bf16_from_fp32_round_to_zero(negativeRoundsUp), 0xBF80);
    EXPECT_EQ(rad::bf16_from_fp32_round_to_nearest_even(negativeRoundsUp), 0xBF81);
    EXPECT_EQ(rad::BFloat16{evenTie}.Bits(), 0x3F80);
    EXPECT_EQ(rad::BFloat16{oddTie}.Bits(), 0x3F82);
    EXPECT_EQ(rad::BFloat16{negativeEvenTie}.Bits(), 0xBF80);
    EXPECT_EQ(rad::BFloat16{negativeOddTie}.Bits(), 0xBF82);
    EXPECT_EQ(rad::BFloat16{std::numeric_limits<float>::max()}.Bits(), 0x7F80);
}

TEST(Core, BFloat16Arithmetic)
{
    const rad::BFloat16 two = 2.0f;
    const rad::BFloat16 four = 4.0f;

    EXPECT_FLOAT_EQ(static_cast<float>(two + four), 6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(four - two), 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(two * four), 8.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(four / two), 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(-two), -2.0f);

    rad::BFloat16 value = two;
    value += four;
    EXPECT_FLOAT_EQ(static_cast<float>(value), 6.0f);
    value -= two;
    EXPECT_FLOAT_EQ(static_cast<float>(value), 4.0f);
    value *= four;
    EXPECT_FLOAT_EQ(static_cast<float>(value), 16.0f);
    value /= two;
    EXPECT_FLOAT_EQ(static_cast<float>(value), 8.0f);

    EXPECT_FLOAT_EQ(two + 0.5f, 2.5f);
    EXPECT_FLOAT_EQ(0.5f + two, 2.5f);
    EXPECT_DOUBLE_EQ(two + 0.5, 2.5);
    EXPECT_DOUBLE_EQ(0.5 - two, -1.5);
    EXPECT_FLOAT_EQ(static_cast<float>(two + 3), 5.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(8 / two), 4.0f);
    EXPECT_TRUE(two < four);
    EXPECT_TRUE(two <= four);
    EXPECT_TRUE(four > two);
    EXPECT_TRUE(four >= two);
    EXPECT_TRUE(two == rad::BFloat16{2.0f});
    EXPECT_TRUE(two != four);
}

TEST(Core, BFloat16SpecialValues)
{
    const rad::BFloat16 positiveZero = rad::BFloat16::FromBits(0x0000);
    const rad::BFloat16 negativeZero = rad::BFloat16::FromBits(0x8000);
    const rad::BFloat16 quietNaN = std::numeric_limits<rad::BFloat16>::quiet_NaN();

    EXPECT_TRUE(positiveZero == negativeZero);
    EXPECT_FALSE(quietNaN == quietNaN);
    EXPECT_TRUE(quietNaN != quietNaN);
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<rad::BFloat16>::min()),
                    std::numeric_limits<float>::min());

    std::ostringstream stream;
    stream << rad::BFloat16{1.5f};
    EXPECT_EQ(stream.str(), "1.5");
}
