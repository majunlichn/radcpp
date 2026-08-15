#include <rad/Core/Sort.h>

#include <gtest/gtest.h>

#include <vector>

using Indices = std::vector<std::size_t>;

TEST(Core, SortIndices)
{
    EXPECT_TRUE(rad::SortIndices(std::vector<int>{}).empty());
    EXPECT_EQ(rad::SortIndices(std::vector<int>{7}), (Indices{0}));
    EXPECT_EQ(rad::SortIndices(std::vector<int>{1, 2, 3}), (Indices{0, 1, 2}));
    EXPECT_EQ(rad::SortIndices(std::vector<int>{4, 3, 2, 1}), (Indices{3, 2, 1, 0}));

    const std::vector<int> values = {3, 1, 4, 5, 2};
    EXPECT_EQ(rad::SortIndices(values), (Indices{1, 4, 0, 2, 3}));
    EXPECT_EQ(rad::SortIndices(values, std::less{}), (Indices{1, 4, 0, 2, 3}));
    EXPECT_EQ(rad::SortIndices(values, std::greater{}), (Indices{3, 2, 0, 4, 1}));
    EXPECT_EQ(values, (std::vector<int>{3, 1, 4, 5, 2}));
}

TEST(Core, StableSortIndices)
{
    EXPECT_TRUE(rad::StableSortIndices(std::vector<int>{}).empty());

    const std::vector<int> values = {3, 1, 2, 1, 2};
    EXPECT_EQ(rad::StableSortIndices(values, std::less{}), (Indices{1, 3, 2, 4, 0}));
    EXPECT_EQ(rad::StableSortIndices(values, std::greater{}), (Indices{0, 2, 4, 1, 3}));
}
