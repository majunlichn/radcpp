#include <rad/Core/Range.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <format>
#include <initializer_list>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

static_assert(std::ranges::view<rad::SliceView<int*>>);
static_assert(std::ranges::random_access_range<rad::SliceView<int*>>);
static_assert(std::ranges::sized_range<rad::SliceView<int*>>);
static_assert(std::ranges::borrowed_range<rad::SliceView<int*>>);

TEST(Core, ToVector)
{
    EXPECT_TRUE(rad::ToVector(std::vector<int>{}).empty());
    EXPECT_EQ(rad::ToVector(std::vector<int>{7}), (std::vector<int>{7}));
    EXPECT_EQ(rad::ToVector(std::array{0, 1, 2, 3}), (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(rad::ToVector(std::views::iota(0, 4)), (std::vector<int>{0, 1, 2, 3}));
}

namespace
{

template <rad::SizedRandomAccessRange R>
std::string RangeToString(R&& range, std::string_view sep = ", ")
{
    if (std::ranges::empty(range))
    {
        return {};
    }

    std::string str;
    if constexpr (std::ranges::sized_range<R>)
    {
        str.reserve(std::ranges::size(range) * (8 + sep.size()));
    }

    auto it = std::ranges::begin(range);
    auto strInserter = std::back_inserter(str);
    std::format_to(strInserter, "{}", *it);
    ++it;
    for (; it != std::ranges::end(range); ++it)
    {
        str += sep;
        std::format_to(strInserter, "{}", *it);
    }
    return str;
}

void TestSlice(const auto& vec, std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step,
               std::initializer_list<int> expected)
{
    const auto sliced = rad::Slice(vec, start, stop, step);
    EXPECT_TRUE(std::ranges::equal(sliced, expected))
        << std::format("vec={{{}}}; start={}; stop={}; step={};\n", RangeToString(vec), start, stop,
                       step)
        << std::format("sliced={{{}}};\n", RangeToString(sliced))
        << std::format("expected={{{}}};\n", RangeToString(expected));
}

} // namespace

TEST(Core, Slice)
{
    { // Syntax and basic functionality tests:
        const std::vector<int> vec = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

        TestSlice(vec, 0, 10, 1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        TestSlice(vec, 2, 8, 1, {2, 3, 4, 5, 6, 7});
        TestSlice(vec, 0, 3, 1, {0, 1, 2});
        TestSlice(vec, 3, 4, 1, {3});
        TestSlice(vec, -3, 10, 1, {7, 8, 9});
        TestSlice(vec, 3, -2, 1, {3, 4, 5, 6, 7});
        TestSlice(vec, -5, -2, 1, {5, 6, 7});
        TestSlice(vec, -100, 100, 1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

        TestSlice(vec, 2, 8, 2, {2, 4, 6});
        TestSlice(vec, 1, 8, 2, {1, 3, 5, 7});
        TestSlice(vec, 0, 10, 2, {0, 2, 4, 6, 8});
        TestSlice(vec, 0, 10, 3, {0, 3, 6, 9});

        TestSlice(vec, 8, 2, -1, {8, 7, 6, 5, 4, 3});
        TestSlice(vec, 8, 2, -2, {8, 6, 4});
        TestSlice(vec, 9, 0, -3, {9, 6, 3});
        TestSlice(vec, -2, 3, -1, {8, 7, 6, 5, 4});
        TestSlice(vec, -1, 0, -1, {9, 8, 7, 6, 5, 4, 3, 2, 1});
        TestSlice(vec, -1, -11, -1, {9, 8, 7, 6, 5, 4, 3, 2, 1, 0});

        TestSlice(vec, 5, 5, 1, {});
        TestSlice(vec, 5, 2, 1, {});
        TestSlice(vec, 2, 5, -1, {});
        TestSlice(vec, 100, 200, 1, {});
        TestSlice(vec, -100, -50, 1, {});
    }

    { // Empty range:
        const std::vector<int> vec;
        TestSlice(vec, 0, 0, 1, {});
        TestSlice(vec, -1, 1, 1, {});
        TestSlice(vec, 0, 0, -1, {});
    }

    { // Single element range:
        const std::vector<int> vec = {7};
        TestSlice(vec, 0, 1, 1, {7});
        TestSlice(vec, -1, 1, 1, {7});
        TestSlice(vec, -1, -2, -1, {7});
        TestSlice(vec, 0, 1, -1, {});
    }

    { // operator->
        struct Element
        {
            int value;
        };
        std::vector<Element> elements = {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};
        const auto sliced = rad::Slice(elements, 0, 10, 1);
        for (auto iter = sliced.begin(); iter != sliced.end(); ++iter)
        {
            iter->value *= 2;
        }
        for (std::size_t i = 0; i < elements.size(); ++i)
        {
            EXPECT_EQ(elements[i].value, static_cast<int>(i * 2));
        }
    }
}
