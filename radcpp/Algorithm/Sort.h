#pragma once

#include <radcpp/Core/Platform.h>
#include <algorithm>
#include <ranges>
#include <vector>
#include <numeric>

namespace rad
{

template<std::ranges::random_access_range Range, typename Compare = std::ranges::less>
std::vector<size_t> SortIndices(const Range& r, Compare comp = {})
{
    std::vector<size_t> indices(std::ranges::size(r));
    std::iota(std::begin(indices), std::end(indices), 0);
    std::ranges::stable_sort(indices,
        [&](size_t i, size_t j) { return comp(r[i], r[j]); });
    return indices;
}

template<std::ranges::random_access_range R>
static inline void SortAscending(R r)
{
    std::sort(std::execution::par_unseq, strs.begin(), strs.end());
}

template<std::ranges::random_access_range R>
static inline void SortDescending(R r)
{
    std::sort(std::execution::par_unseq, strs.begin(), strs.end(), std::greater<T>());
}

} // namespace rad
