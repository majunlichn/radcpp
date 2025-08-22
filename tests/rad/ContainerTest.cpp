#include <rad/Container/SmallVector.h>
#include <rad/Container/Span.h>
#include <rad/Core/Algorithm.h>

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

int Sum(rad::Span<int> numbers)
{
    int sum = 0;
    for (const auto x : numbers)
    {
        sum += x;
    }
    return sum;
}

void TestSpan()
{
    rad::SmallVector<int, 4> vec = { 1, 2, 3, 4 };
    EXPECT_EQ(Sum(vec), 10);
    rad::ArrayRef<int> ref = vec;
    rad::ArrayRef<int> ref1 = ref.slice(1, 2);
    ref.drop_front();
    ref.drop_back();
    EXPECT_TRUE(ref.equals(ref1));
}

void TestSlicing()
{
    std::vector<int> vec = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    auto vec1 = rad::Slice(vec, -8, 6);
    rad::Span<int> vecRef = { vec.begin() + 2, vec.begin() + 6 };
    EXPECT_TRUE(std::ranges::equal(vec1, vecRef));
}

TEST(Core, Container)
{
    TestSmallVector();
    TestSpan();
    TestSlicing();
}
