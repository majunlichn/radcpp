#pragma once

#include <concepts>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

namespace rad
{

// A range that knows its size in constant time and allows random access to its elements.
// Matches the C++26 exposition-only concept sized-random-access-range.
template <class T>
concept SizedRandomAccessRange =
    std::ranges::random_access_range<T> && std::ranges::sized_range<T>;

// Copies the elements of r into a std::vector.
template <std::ranges::input_range R>
    requires std::constructible_from<std::ranges::range_value_t<R>, std::ranges::range_reference_t<R>>
[[nodiscard]] std::vector<std::ranges::range_value_t<R>> ToVector(R&& r)
{
    std::vector<std::ranges::range_value_t<R>> result;
    if constexpr (std::ranges::sized_range<R>)
    {
        result.reserve(static_cast<std::size_t>(std::ranges::size(r)));
    }
    for (auto&& elem : r)
    {
        result.emplace_back(std::forward<decltype(elem)>(elem));
    }
    return result;
}

} // namespace rad
