#include <rad/Core/Crc.h>

#include <gtest/gtest.h>

#include <string_view>

TEST(Core, Crc)
{
    constexpr std::string_view data = "123456789";
    EXPECT_EQ(rad::Crc32::Compute(data), 0xCBF43926u);
}
