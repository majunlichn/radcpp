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

} // namespace rad
