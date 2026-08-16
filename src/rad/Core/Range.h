#pragma once

#include <ranges>

namespace rad
{

// A range that knows its size in constant time and allows random access to its elements.
// Matches the C++26 exposition-only concept sized-random-access-range.
template <class T>
concept SizedRandomAccessRange =
    std::ranges::random_access_range<T> && std::ranges::sized_range<T>;

} // namespace rad
