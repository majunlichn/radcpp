#pragma once

#include <rad/Core/Range.h>

#include <algorithm>
#include <cstddef>
#include <execution>
#include <numeric>
#include <ranges>
#include <vector>

namespace rad
{

// Indices that would sort r (NumPy argsort). Does not modify r.
template <SizedRandomAccessRange Range, typename Compare = std::ranges::less,
          typename ExecutionPolicy = std::execution::sequenced_policy>
[[nodiscard]] std::vector<std::size_t> SortIndices(Range&& r, Compare comp = {},
                                                   ExecutionPolicy policy = std::execution::seq)
{
    const auto count = std::ranges::size(r);
    if (count == 0)
    {
        return {};
    }

    std::vector<std::size_t> indices(count);
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    auto it = std::ranges::begin(r);
    std::sort(policy, indices.begin(), indices.end(),
              [&](std::size_t i, std::size_t j) { return comp(it[i], it[j]); });
    return indices;
}

// Like SortIndices, but equal elements keep their original relative order.
template <SizedRandomAccessRange Range, typename Compare = std::ranges::less,
          typename ExecutionPolicy = std::execution::sequenced_policy>
[[nodiscard]] std::vector<std::size_t> StableSortIndices(
    Range&& r, Compare comp = {}, ExecutionPolicy policy = std::execution::seq)
{
    const auto count = std::ranges::size(r);
    if (count == 0)
    {
        return {};
    }

    std::vector<std::size_t> indices(count);
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    auto it = std::ranges::begin(r);
    std::stable_sort(policy, indices.begin(), indices.end(),
                     [&](std::size_t i, std::size_t j) { return comp(it[i], it[j]); });
    return indices;
}

} // namespace rad
