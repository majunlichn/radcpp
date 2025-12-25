#pragma once

#include <rad/Common/Platform.h>

namespace rad
{

namespace detail {

struct ignore_t
{
    template <typename T>
    constexpr void operator=(T&&) const noexcept
    {
    }
};

} // namespace detail

inline constexpr detail::ignore_t ignore;

} // namespace rad
