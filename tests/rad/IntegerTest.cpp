#include <rad/Core/Integer.h>

#include <gtest/gtest.h>

TEST(Core, Integer)
{
    EXPECT_EQ(rad::BitScanReverse32(0x00000001u), 0);
    EXPECT_EQ(rad::BitScanReverse32(0x00008001u), 15);
    EXPECT_EQ(rad::BitScanReverse32(0x00010001u), 16);
    EXPECT_EQ(rad::BitScanReverse32(0x80000001u), 31);
    EXPECT_EQ(rad::BitScanReverse64(0x0000000000000001ull), 0);
    EXPECT_EQ(rad::BitScanReverse64(0x0000000080000001ull), 31);
    EXPECT_EQ(rad::BitScanReverse64(0x0000000100000001ull), 32);
    EXPECT_EQ(rad::BitScanReverse64(0x8000000000000001ull), 63);

    EXPECT_EQ(rad::CountBits32(0x00000000u), 0);
    EXPECT_EQ(rad::CountBits32(0x55555555u), 16);
    EXPECT_EQ(rad::CountBits32(0xAAAAAAAAu), 16);
    EXPECT_EQ(rad::CountBits32(0xFFFFFFFFu), 32);
    EXPECT_EQ(rad::CountBits64(0x0000000000000000ull), 0);
    EXPECT_EQ(rad::CountBits64(0x5555555555555555ull), 32);
    EXPECT_EQ(rad::CountBits64(0xAAAAAAAAAAAAAAAAull), 32);
    EXPECT_EQ(rad::CountBits64(0xFFFFFFFFFFFFFFFFull), 64);
}
