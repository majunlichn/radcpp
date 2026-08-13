#pragma once

#include <boost/outcome/result.hpp>

#include <system_error>
#include <utility>

namespace rad
{

template <typename T, typename E = std::error_code>
using Result = BOOST_OUTCOME_V2_NAMESPACE::result<T, E>;

template <typename T>
[[nodiscard]] constexpr auto Success(T&& value)
{
    return BOOST_OUTCOME_V2_NAMESPACE::success(std::forward<T>(value));
}

[[nodiscard]] constexpr auto Success() noexcept
{
    return BOOST_OUTCOME_V2_NAMESPACE::success();
}

template <typename E>
[[nodiscard]] constexpr auto Failure(E&& error)
{
    return BOOST_OUTCOME_V2_NAMESPACE::failure(std::forward<E>(error));
}

[[nodiscard]] inline auto Failure(std::errc error) noexcept
{
    return BOOST_OUTCOME_V2_NAMESPACE::failure(std::make_error_code(error));
}

} // namespace rad
