#pragma once

#include <concepts>
#include <type_traits>

namespace rad
{

template <class T, class... Types>
constexpr bool is_any_of = (std::is_same_v<T, Types> || ...);

template <typename T, typename... Types>
constexpr bool are_all_same = (std::is_same_v<T, Types> && ...);

template <class T>
concept Enumeration = std::is_enum_v<T>;

template <class T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

template <Enumeration T>
constexpr auto UnderlyingCast(T t) noexcept
{
    return static_cast<std::underlying_type_t<T>>(t);
}

} // namespace rad
