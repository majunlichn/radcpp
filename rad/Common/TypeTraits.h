#pragma once

#include <rad/Common/Platform.h>
#include <cstring>
#include <concepts>

namespace rad
{

template <typename...>
struct Sizeof;

template <>
struct Sizeof<>
{
    static constexpr uint32_t value = 0;
};

template <typename T, typename... Rest>
struct Sizeof<T, Rest...>
{
    static constexpr uint32_t value = sizeof(T) + Sizeof<Rest...>::value;
};

template <typename...>
struct MaxSizeof;

template <>
struct MaxSizeof<>
{
    static constexpr uint32_t value = 0;
};

template <typename T, typename... Rest>
struct MaxSizeof<T, Rest...>
{
    static constexpr uint32_t value = sizeof(T) > MaxSizeof<Rest...>::value ?
        sizeof(T) : MaxSizeof<Rest...>::value;
};

template <class T>
concept Enumeration = std::is_enum_v<T>;

template <class T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

template<class T>
constexpr auto ToUnderlying(T t) noexcept
{
    return static_cast<std::underlying_type_t<T>>(t);
}

// Implementations of std::bit_cast() from C++ 20.
// Please refer to: https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
std::enable_if_t<
    sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From>&&
    std::is_trivially_copyable_v<To>,
    To>
    // constexpr support needs compiler magic
    bit_cast(const From& src) noexcept
{
    static_assert(
        std::is_trivially_constructible_v<To>,
        "destination type must be trivially constructible");

    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

} // namespace rad
