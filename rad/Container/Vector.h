#pragma once

#include <rad/Core/Platform.h>
#include <ranges>
#include <vector>

namespace rad
{

template<typename T, std::ranges::sized_range Range>
std::vector<T> ToVec(Range r)
{
    std::vector<T> vec;
    vec.reserve(std::ranges::size(r));
    for (const auto& e : r)
    {
        vec.push_back(static_cast<T>(e));
    }
    return vec;
}

template<typename T>
static inline void RemoveDuplicated(std::vector<T>& elems)
{
    std::sort(std::execution::par_unseq, elems.begin(), elems.end());
    auto iter = std::unique(elems.begin(), elems.end());
    elems.erase(iter, elems.end());
}

} // namespace rad
