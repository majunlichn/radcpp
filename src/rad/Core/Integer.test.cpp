#include <rad/Core/Integer.h>
#include <rad/Core/IntegerFallback.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

static_assert(rad::Integer<int>);
static_assert(rad::Integer<const unsigned>);
static_assert(!rad::Integer<bool>);
static_assert(!rad::Integer<const bool>);
static_assert(rad::UnsignedInteger<unsigned>);
static_assert(!rad::UnsignedInteger<int>);
static_assert(!rad::UnsignedInteger<bool>);
static_assert(rad::BitCount<std::uint32_t> == 32);
static_assert(rad::BitCount<std::uint64_t> == 64);

TEST(Core, IntegerBitOperations)
{
    EXPECT_FALSE(rad::IsPowerOfTwo(0u));
    EXPECT_TRUE(rad::IsPowerOfTwo(1u));
    EXPECT_TRUE(rad::IsPowerOfTwo(64u));
    EXPECT_FALSE(rad::IsPowerOfTwo(65u));

    EXPECT_FALSE(rad::IsPowerOfTwo(-8));
    EXPECT_FALSE(rad::IsPowerOfTwo(0));
    EXPECT_TRUE(rad::IsPowerOfTwo(8));

    EXPECT_TRUE(rad::HasBits(0b1110u, 0b0110u));
    EXPECT_TRUE(rad::HasAnyBits(0b1000u, 0b1010u));
    EXPECT_TRUE(rad::HasNoBits(0b1000u, 0b0110u));

    EXPECT_EQ(rad::Upper16Bits(0x12345678u), 0x1234u);
    EXPECT_EQ(rad::Lower16Bits(0x12345678u), 0x5678u);
    EXPECT_EQ(rad::Upper16Bits(0x0000FFFFu), 0x0000u);
    EXPECT_EQ(rad::Lower16Bits(0x0000FFFFu), 0xFFFFu);
    EXPECT_EQ(rad::Upper32Bits(0x123456789ABCDEF0ull), 0x12345678u);
    EXPECT_EQ(rad::Lower32Bits(0x123456789ABCDEF0ull), 0x9ABCDEF0u);
    EXPECT_EQ(rad::Upper32Bits(0x00000000FFFFFFFFull), 0x00000000u);
    EXPECT_EQ(rad::Lower32Bits(0x00000000FFFFFFFFull), 0xFFFFFFFFu);

    EXPECT_EQ(rad::BitScanReverse(0x80u), 7);
    EXPECT_EQ(rad::CountSetBits(0b10110100u), 4);
    EXPECT_EQ(rad::ReverseBits<std::uint8_t>(0), 0);
    EXPECT_EQ(rad::ReverseBits<std::uint8_t>(0b00010010), 0b01001000);
}

TEST(Core, IntegerBitRanges)
{
    EXPECT_EQ(rad::SignExtend<std::uint8_t>(0b0111, 4), 7);
    EXPECT_EQ(rad::SignExtend<std::uint8_t>(0b1111, 4), -1);
    EXPECT_EQ(rad::SignExtend<std::uint8_t>(0b11110111, 4), 7);
    EXPECT_EQ(rad::SignExtend<std::uint8_t>(0b10000000, 8), -128);
    EXPECT_EQ(rad::SignExtend<std::uint32_t>(0x80000000u, 32),
              std::numeric_limits<std::int32_t>::min());
    EXPECT_EQ(rad::SignExtend<std::uint64_t>(0x8000000000000000ull, 64),
              std::numeric_limits<std::int64_t>::min());

    EXPECT_EQ(rad::ExtractBits(0b110110u, 1, 3), 0b011u);
    EXPECT_EQ(rad::ExtractBits(0b110110u, 0, 0), 0u);
    EXPECT_EQ(rad::ExtractBits(0b110110u, 0, rad::BitCount<unsigned>), 0b110110u);

    EXPECT_EQ(rad::ReplaceBits(0b11110000u, 0b101u, 1, 3), 0b11111010u);
    EXPECT_EQ(rad::ReplaceBits(0b11110000u, 0u, 1, 0), 0b11110000u);
    EXPECT_EQ(rad::ReplaceBits(0b11110000u, 0b10101010u, 0, rad::BitCount<unsigned>), 0b10101010u);
}

TEST(Core, IntegerCheckedArithmetic)
{
    EXPECT_FALSE(rad::AddWouldOverflow(20, 22));
    EXPECT_TRUE(rad::AddWouldOverflow(std::numeric_limits<int>::max(), 1));
    EXPECT_TRUE(rad::AddWouldOverflow(std::numeric_limits<int>::min(), -1));
    EXPECT_TRUE(rad::AddWouldOverflow<std::uint8_t>(250, 6));

    EXPECT_EQ(rad::CheckedAdd(20, 22), 42);
    EXPECT_EQ(rad::CheckedAdd(-20, -22), -42);
    EXPECT_FALSE(rad::CheckedAdd(std::numeric_limits<int>::max(), 1).has_value());
    EXPECT_FALSE(rad::CheckedAdd(std::numeric_limits<int>::min(), -1).has_value());

    EXPECT_EQ(rad::CheckedAdd<std::uint8_t>(250, 5), 255);
    EXPECT_FALSE(rad::CheckedAdd<std::uint8_t>(250, 6).has_value());

    EXPECT_FALSE(rad::SubtractWouldOverflow(50, 8));
    EXPECT_TRUE(rad::SubtractWouldOverflow(std::numeric_limits<int>::min(), 1));
    EXPECT_TRUE(rad::SubtractWouldOverflow(std::numeric_limits<int>::max(), -1));
    EXPECT_TRUE(rad::SubtractWouldOverflow<std::uint8_t>(40, 42));

    EXPECT_EQ(rad::CheckedSubtract(50, 8), 42);
    EXPECT_EQ(rad::CheckedSubtract(-50, -8), -42);
    EXPECT_FALSE(rad::CheckedSubtract(std::numeric_limits<int>::min(), 1).has_value());
    EXPECT_FALSE(rad::CheckedSubtract(std::numeric_limits<int>::max(), -1).has_value());

    EXPECT_EQ(rad::CheckedSubtract<std::uint8_t>(42, 40), 2);
    EXPECT_FALSE(rad::CheckedSubtract<std::uint8_t>(40, 42).has_value());
}

TEST(Core, IntegerRoundingAndAlignment)
{
    EXPECT_EQ(rad::DivideRoundUp(0u, 4u), 0u);
    EXPECT_EQ(rad::DivideRoundUp(1u, 4u), 1u);
    EXPECT_EQ(rad::DivideRoundUp(8u, 4u), 2u);
    EXPECT_EQ(rad::DivideRoundUp(9u, 4u), 3u);

    EXPECT_EQ(rad::AlignUpToPowerOfTwo(0u, 8u), 0u);
    EXPECT_EQ(rad::AlignUpToPowerOfTwo(17u, 8u), 24u);
    EXPECT_EQ(rad::AlignDownToPowerOfTwo(23u, 8u), 16u);
    EXPECT_EQ(rad::AlignUpToPowerOfTwo(16u, 8u), 16u);
    EXPECT_EQ(rad::AlignDownToPowerOfTwo(16u, 8u), 16u);

    EXPECT_EQ(rad::RoundDownToMultiple(17u, 8u), 16u);
    EXPECT_EQ(rad::RoundDownToMultiple(16u, 8u), 16u);
    EXPECT_EQ(rad::RoundDownToMultiple(17u, 1u), 17u);

    EXPECT_EQ(rad::RoundUpToMultiple(0u, 8u), 0u);
    EXPECT_EQ(rad::RoundUpToMultiple(17u, 8u), 24u);
    EXPECT_EQ(rad::RoundUpToMultiple(16u, 8u), 16u);
    EXPECT_EQ(rad::RoundUpToMultiple(17u, 1u), 17u);

    EXPECT_EQ(rad::RoundUpToNextPowerOfTwo(0u), 1u);
    EXPECT_EQ(rad::RoundUpToNextPowerOfTwo(1u), 2u);
    EXPECT_EQ(rad::RoundUpToNextPowerOfTwo(16u), 32u);

    EXPECT_EQ(rad::RoundUpToPowerOfTwo(0u), 1u);
    EXPECT_EQ(rad::RoundUpToPowerOfTwo(1u), 1u);
    EXPECT_EQ(rad::RoundUpToPowerOfTwo(17u), 32u);
    EXPECT_EQ(rad::RoundUpToPowerOfTwo<std::uint8_t>(128), 128);

    EXPECT_EQ(rad::RoundDownToPowerOfTwo(0u), 0u);
    EXPECT_EQ(rad::RoundDownToPowerOfTwo(17u), 16u);
}

TEST(Core, IntegerFallback)
{
    EXPECT_EQ(rad::fallback::BitScanReverse(std::uint8_t{0x80}),
              rad::BitScanReverse(std::uint8_t{0x80}));
    EXPECT_EQ(rad::fallback::BitScanReverse(0x80000000u), rad::BitScanReverse(0x80000000u));
    EXPECT_EQ(rad::fallback::BitScanReverse(0x8000000000000000ull),
              rad::BitScanReverse(0x8000000000000000ull));

    EXPECT_EQ(rad::fallback::CountSetBits(std::uint8_t{0b10110100}),
              rad::CountSetBits(std::uint8_t{0b10110100}));
    EXPECT_EQ(rad::fallback::CountSetBits(0xF0F0F00Fu), rad::CountSetBits(0xF0F0F00Fu));
    EXPECT_EQ(rad::fallback::CountSetBits(0xF0F0F0F00F0F0F0Full),
              rad::CountSetBits(0xF0F0F0F00F0F0F0Full));

    EXPECT_EQ(rad::fallback::ReverseBits(std::uint8_t{0b00010010}),
              rad::ReverseBits(std::uint8_t{0b00010010}));
    EXPECT_EQ(rad::fallback::ReverseBits(0x12345678u), 0x1E6A2C48u);
    EXPECT_EQ(rad::fallback::ReverseBits(0x123456789ABCDEF0ull), 0x0F7B3D591E6A2C48ull);

    EXPECT_EQ(rad::fallback::RoundUpToNextPowerOfTwo(0u), rad::RoundUpToNextPowerOfTwo(0u));
    EXPECT_EQ(rad::fallback::RoundUpToNextPowerOfTwo(16u), rad::RoundUpToNextPowerOfTwo(16u));
    EXPECT_EQ(rad::fallback::RoundUpToPowerOfTwo(17u), rad::RoundUpToPowerOfTwo(17u));
    EXPECT_EQ(rad::fallback::RoundUpToPowerOfTwo<std::uint8_t>(128),
              rad::RoundUpToPowerOfTwo<std::uint8_t>(128));
    EXPECT_EQ(rad::fallback::RoundDownToPowerOfTwo(0u), rad::RoundDownToPowerOfTwo(0u));
    EXPECT_EQ(rad::fallback::RoundDownToPowerOfTwo(17u), rad::RoundDownToPowerOfTwo(17u));
}

TEST(Core, IntegerLiterals)
{
    using namespace rad::literals;

    EXPECT_EQ(2_KiB, 2048ull);
    EXPECT_EQ(3_MiB, 3ull * 1024ull * 1024ull);
    EXPECT_EQ(4_GiB, 4ull * 1024ull * 1024ull * 1024ull);
    EXPECT_EQ(5_TiB, 5ull * 1024ull * 1024ull * 1024ull * 1024ull);
    EXPECT_EQ(6_PiB, 6ull * 1024ull * 1024ull * 1024ull * 1024ull * 1024ull);
    EXPECT_EQ(7_EiB, 7ull * 1024ull * 1024ull * 1024ull * 1024ull * 1024ull * 1024ull);
}
