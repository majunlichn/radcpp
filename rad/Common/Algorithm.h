#pragma once

#include <rad/Common/Platform.h>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

namespace rad
{

// Support Python like list slicing.
template<std::ranges::random_access_range R>
R Slice(const R& vec, std::ptrdiff_t begin, std::ptrdiff_t end)
{
    if (begin < 0)
    {
        begin = vec.size() + begin;
        if (begin < 0)
        {
            begin = 0;
        }
    }

    if (end < 0)
    {
        end = vec.size() + end;
        if (end < 0)
        {
            end = 0;
        }
    }

    if (begin >= end)
    {
        return {};
    }

    if (end < vec.size())
    {
        return R(vec.begin() + begin, vec.begin() + end);
    }
    else
    {
        return R(vec.begin() + begin, vec.end());
    }
}

template<std::ranges::random_access_range R>
static inline void SortAscending(R& r)
{
    std::ranges::sort(r, std::ranges::less{});
}

template<std::ranges::random_access_range R>
static inline void SortDescending(R& r)
{
    std::ranges::sort(r, std::ranges::greater{});
}

template<std::ranges::random_access_range Range, typename Compare = std::ranges::less>
std::vector<size_t> SortIndices(const Range& r, Compare comp = {})
{
    std::vector<size_t> indices(std::ranges::size(r));
    std::iota(std::begin(indices), std::end(indices), 0);
    std::ranges::stable_sort(indices,
        [&](size_t i, size_t j) { return comp(r[i], r[j]); });
    return indices;
}

} // namespace rad
