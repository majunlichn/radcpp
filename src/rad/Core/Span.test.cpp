#include <rad/Core/Span.h>

#include <gtest/gtest.h>

#include <array>
#include <span>
#include <type_traits>
#include <vector>

static_assert(std::is_trivially_copyable_v<rad::Span<int>>);
static_assert(std::is_constructible_v<rad::Span<int, 1>, int&>);
static_assert(!std::is_constructible_v<rad::Span<int, 2>, int&>);
static_assert(!std::is_constructible_v<rad::Span<int>, int>);
static_assert(std::is_constructible_v<rad::Span<const int>, int>);
static_assert(!std::is_constructible_v<rad::Span<int>, std::initializer_list<int>>);
static_assert(std::is_constructible_v<rad::Span<const int>, std::initializer_list<int>>);
static_assert(!std::is_constructible_v<rad::Span<int>, std::vector<int>>);
static_assert(std::is_constructible_v<rad::Span<const int>, std::vector<int>>);

TEST(Core, Span)
{
    // Single elements can adapt to mutable, const, and fixed extent spans.
    {
        int value = 7;
        rad::Span<int> mutableSpan(value);
        static_assert(std::same_as<decltype(mutableSpan), rad::Span<int>>);

        ASSERT_EQ(mutableSpan.size(), 1);
        EXPECT_EQ(mutableSpan.data(), &value);

        mutableSpan[0] = 42;
        EXPECT_EQ(value, 42);

        const int constValue = 5;
        rad::Span constSpan(constValue);
        static_assert(std::same_as<decltype(constSpan), rad::Span<const int>>);

        ASSERT_EQ(constSpan.size(), 1);
        EXPECT_EQ(constSpan.data(), &constValue);
        EXPECT_EQ(constSpan[0], 5);

        rad::Span<int, 1> fixedExtentSpan(value);
        ASSERT_EQ(fixedExtentSpan.size(), 1);
        EXPECT_EQ(fixedExtentSpan.data(), &value);
    }

    // Contiguous ranges preserve mutability for lvalues and become const for temporaries.
    {
        std::vector<int> values = {1, 2, 3};
        rad::Span mutableSpan(values);
        static_assert(std::same_as<decltype(mutableSpan), rad::Span<int>>);

        ASSERT_EQ(mutableSpan.size(), values.size());
        EXPECT_EQ(mutableSpan.data(), values.data());

        mutableSpan[0] = 4;
        EXPECT_EQ(values[0], 4);

        const std::vector<int> constValues = {5, 6, 7};
        rad::Span constSpan(constValues);
        static_assert(std::same_as<decltype(constSpan), rad::Span<const int>>);

        ASSERT_EQ(constSpan.size(), constValues.size());
        EXPECT_EQ(constSpan.data(), constValues.data());
        EXPECT_EQ(constSpan[0], 5);

        static_assert(
            std::same_as<decltype(rad::Span(std::vector<int>{8, 9})), rad::Span<const int>>);

        auto first = [](rad::Span<const int> span) { return span[0]; };
        EXPECT_EQ(first(std::vector<int>{8, 9}), 8);
    }

    // Initializer lists can bind to const spans for immediate function arguments.
    {
        auto sum = [](rad::Span<const int> values)
        {
            int result = 0;
            for (const int value : values)
            {
                result += value;
            }
            return result;
        };

        EXPECT_EQ(sum({1, 2, 3}), 6);
    }

    // MakeSpan adapts arrays, pointer ranges, single elements, and const ranges.
    {
        int values[] = {1, 2, 3};

        const auto arraySpan = rad::MakeSpan(values);
        static_assert(std::same_as<decltype(arraySpan), const std::span<int>>);
        EXPECT_EQ(arraySpan.data(), values);
        EXPECT_EQ(arraySpan.size(), 3);

        const auto pointerSizeSpan = rad::MakeSpan(values, 2);
        EXPECT_EQ(pointerSizeSpan.data(), values);
        EXPECT_EQ(pointerSizeSpan.size(), 2);

        const auto pointerRangeSpan = rad::MakeSpan(values, values + 3);
        EXPECT_EQ(pointerRangeSpan.data(), values);
        EXPECT_EQ(pointerRangeSpan.size(), 3);

        int value = 9;
        auto singleSpan = rad::MakeSpan(value);
        static_assert(std::same_as<decltype(singleSpan), std::span<int>>);
        ASSERT_EQ(singleSpan.size(), 1);
        singleSpan[0] = 10;
        EXPECT_EQ(value, 10);

        const std::array constValues = {4, 5, 6};
        auto constRangeSpan = rad::MakeSpan(constValues);
        static_assert(std::same_as<decltype(constRangeSpan), std::span<const int>>);
        EXPECT_EQ(constRangeSpan.data(), constValues.data());
        EXPECT_EQ(constRangeSpan.size(), constValues.size());
    }
}
