#include <rad/Common/Iterator.h>
#include <algorithm>
#include <execution>

#include <gtest/gtest.h>

#include <concepts>

TEST(Core, Iterator)
{
    constexpr size_t count = 4;
    int input[count] = { 1, 2, 3, 4 };
    int output[count];
    auto beg = rad::MakeZipPointer(std::begin(input), std::begin(output));
    auto end = rad::MakeZipPointer(std::end(input), std::end(output));

    std::for_each(std::execution::par_unseq,
        beg, end, [](auto t) {
            std::get<1>(t) = (std::get<0>(t) * 2);
        });

    for (int i = 0; i < count; i++)
    {
        EXPECT_TRUE(output[i] == (input[i] * 2));
    }
}
