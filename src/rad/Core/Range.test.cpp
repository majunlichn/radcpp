#include <rad/Core/Range.h>

#include <gtest/gtest.h>

#include <array>
#include <string_view>
#include <vector>

TEST(Core, ToVector)
{
    EXPECT_TRUE(rad::ToVector(std::vector<int>{}).empty());
    EXPECT_EQ(rad::ToVector(std::vector<int>{7}), (std::vector<int>{7}));
    EXPECT_EQ(rad::ToVector(std::array{0, 1, 2, 3}), (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(rad::ToVector(std::views::iota(0, 4)), (std::vector<int>{0, 1, 2, 3}));
}
