#pragma once

#include <cassert>

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <ranges>
#include <span>
#include <type_traits>

namespace rad
{
namespace detail
{

template <typename From, typename To>
concept SpanCompatibleElement = std::convertible_to<From (*)[], To (*)[]>;

template <typename R, typename T>
concept SpanCompatibleRange =
    std::ranges::contiguous_range<R> &&
    (std::is_lvalue_reference_v<R> || std::ranges::borrowed_range<R> || std::is_const_v<T>) &&
    requires(R&& range) {
        { std::ranges::data(range) } -> std::convertible_to<T*>;
    };

template <std::ranges::contiguous_range R>
using SpanRangeElement = std::remove_reference_t<std::ranges::range_reference_t<R>>;

template <std::ranges::contiguous_range R>
using SpanDeducedRangeElement =
    std::conditional_t<std::is_lvalue_reference_v<R> || std::ranges::borrowed_range<R>,
                       SpanRangeElement<R>, const SpanRangeElement<R>>;

} // namespace detail

// std::span extension with implicit adapters for function parameters (initializer_list,
// contiguous ranges, single elements, etc.). Not safe to store a Span; lifetime is bound
// to the caller's argument. Trivially copyable; pass by value.
template <typename T, std::size_t Extent = std::dynamic_extent>
class [[nodiscard]] Span : public std::span<T, Extent>
{
public:
    using SpanBase = std::span<T, Extent>;

    using SpanBase::span;

    constexpr Span(std::span<T, Extent> s) :
        SpanBase(s)
    {
    }

    template <typename U>
        requires((Extent == std::dynamic_extent || Extent == 1) &&
                 !std::ranges::contiguous_range<U> && detail::SpanCompatibleElement<U, T>)
    constexpr Span(U& elem) :
        SpanBase(&elem, 1)
    {
    }

    template <typename U>
        requires((Extent == std::dynamic_extent || Extent == 1) &&
                 !std::ranges::contiguous_range<U> && detail::SpanCompatibleElement<const U, T>)
    constexpr Span(const U& elem) :
        SpanBase(&elem, 1)
    {
    }

    constexpr Span(std::initializer_list<std::remove_const_t<T>> list)
        requires(Extent == std::dynamic_extent && std::is_const_v<T>)
        :
        SpanBase(list.begin(), list.size())
    {
    }

    template <std::ranges::contiguous_range R>
        requires(Extent == std::dynamic_extent &&
                 !std::same_as<std::remove_cvref_t<R>, Span<T, Extent>> &&
                 detail::SpanCompatibleRange<R, T>)
    constexpr Span(R&& range) :
        SpanBase(std::ranges::data(range), std::ranges::size(range))
    {
    }
}; // class Span

template <typename T>
Span(std::initializer_list<T> list) -> Span<const T>;

template <typename T>
    requires(!std::ranges::contiguous_range<T>)
Span(T& elem) -> Span<T>;

template <typename T>
    requires(!std::ranges::contiguous_range<T>)
Span(const T& elem) -> Span<const T>;

template <std::ranges::contiguous_range R>
Span(R&& range) -> Span<detail::SpanDeducedRangeElement<R>>;

template <typename T, std::size_t N>
constexpr auto MakeSpan(T (&arr)[N])
{
    return std::span<T>(arr, N);
}

// Accepts both lvalues and rvalues.
// If R is an rvalue (temporary) and not a explicitly borrowed range, forces the Span to be const T.
template <std::ranges::contiguous_range R>
constexpr auto MakeSpan(R&& r)
{
    using ElementType = std::remove_reference_t<std::ranges::range_reference_t<R>>;

    // If the range is a temporary, force 'const' so you can't mutate a dying object.
    using T = std::conditional_t<std::is_lvalue_reference_v<R> || std::ranges::borrowed_range<R>,
                                 ElementType, const ElementType>;

    return std::span<T>{std::ranges::data(r), std::ranges::size(r)};
}

template <typename T>
constexpr auto MakeSpan(const std::initializer_list<T>& list)
{
    return std::span<const T>(list.begin(), list.size());
}

template <typename T>
    requires(!std::ranges::contiguous_range<T>)
constexpr auto MakeSpan(T& element)
{
    return std::span<T>(&element, 1);
}

// Binds to single const lvalues OR single temporaries (e.g. MakeSpan(42)).
// Temporaries will naturally become std::span<const T>.
template <typename T>
    requires(!std::ranges::contiguous_range<T>)
constexpr auto MakeSpan(const T& element)
{
    return std::span<const T>(&element, 1);
}

template <typename T>
constexpr auto MakeSpan(T* ptr, std::size_t size)
{
    return std::span<T>(ptr, size);
}

template <typename T>
constexpr auto MakeSpan(T* begin, T* end)
{
    assert(begin <= end);
    return std::span<T>(begin, static_cast<std::size_t>(end - begin));
}

} // namespace rad
