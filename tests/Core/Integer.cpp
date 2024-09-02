#include <rad/Core/Integer.h>
#include <gtest/gtest.h>

TEST(Core, Integer)
{
    EXPECT_EQ(rad::BitScanReverse32(0x00000001u), 0);
    EXPECT_EQ(rad::BitScanReverse32(0x00010002u), 16);
    EXPECT_EQ(rad::BitScanReverse32(0x00080003u), 19);
    EXPECT_EQ(rad::BitScanReverse32(0x80000004u), 31);
    EXPECT_EQ(rad::BitScanReverse64(0x0000000000000001ull), 0);
    EXPECT_EQ(rad::BitScanReverse64(0x0000000100000002ull), 32);
    EXPECT_EQ(rad::BitScanReverse64(0x0000000800000003ull), 35);
    EXPECT_EQ(rad::BitScanReverse64(0x8000000000000004ull), 63);

    EXPECT_EQ(rad::CountBits(0x00000000u), 0);
    EXPECT_EQ(rad::CountBits(0x55555555u), 16);
    EXPECT_EQ(rad::CountBits(0xAAAAAAAAu), 16);
    EXPECT_EQ(rad::CountBits(0xFFFFFFFFu), 32);
    EXPECT_EQ(rad::CountBits(0x0000000000000000ull), 0);
    EXPECT_EQ(rad::CountBits(0x5555555555555555ull), 32);
    EXPECT_EQ(rad::CountBits(0xAAAAAAAAAAAAAAAAull), 32);
    EXPECT_EQ(rad::CountBits(0xFFFFFFFFFFFFFFFFull), 64);
}
