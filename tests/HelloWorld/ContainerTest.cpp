#include <rad/Container/SmallVector.h>

#include <gtest/gtest.h>

void TestSmallVector()
{
    rad::SmallVector<int, 8> vec;
    vec = { 1, 2, 3, 4 };
    vec.push_back(5);
    vec.push_back(6);
    vec.push_back(7);
    vec.push_back(8);
    EXPECT_EQ(vec.size(), 8);
    vec.push_back(9);
    vec.push_back(10);
    vec.push_back(11);
    vec.push_back(12);
    EXPECT_EQ(vec.size(), 12);
    for (size_t i = 0; i < vec.size(); ++i)
    {
        EXPECT_EQ(vec[i], i + 1);
    }
}

TEST(Core, Container)
{
    TestSmallVector();
}
