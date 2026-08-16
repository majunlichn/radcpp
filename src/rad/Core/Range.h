#pragma once

#include <cassert>
#include <compare>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace rad
{

// A range that knows its size in constant time and allows random access to its elements.
// Matches the C++26 exposition-only concept sized-random-access-range.
template <class T>
concept SizedRandomAccessRange = std::ranges::random_access_range<T> && std::ranges::sized_range<T>;

// Copies the elements of r into a std::vector.
template <std::ranges::input_range R>
    requires std::constructible_from<std::ranges::range_value_t<R>,
                                     std::ranges::range_reference_t<R>>
[[nodiscard]] std::vector<std::ranges::range_value_t<R>> ToVector(R&& r)
{
    std::vector<std::ranges::range_value_t<R>> result;
    if constexpr (std::ranges::sized_range<R>)
    {
        result.reserve(static_cast<std::size_t>(std::ranges::size(r)));
    }
    for (auto&& elem : r)
    {
        result.emplace_back(std::forward<decltype(elem)>(elem));
    }
    return result;
}

namespace detail
{

// Python slice.indices / PySlice_AdjustIndices. start and stop are inclusive/exclusive
// indices that may be negative or past either end. Returns the number of selected elements
// and rewrites start to the first selected index when the result is non-empty.
// https://github.com/python/cpython/blob/main/Objects/sliceobject.c
[[nodiscard]] constexpr std::ptrdiff_t NormalizeSlice(std::ptrdiff_t length, std::ptrdiff_t& start,
                                                      std::ptrdiff_t& stop,
                                                      std::ptrdiff_t step) noexcept
{
    assert(length >= 0);
    assert(step != 0);

    if (start < 0)
    {
        start += length;
        if (start < 0)
        {
            start = (step < 0) ? -1 : 0;
        }
    }
    else if (start >= length)
    {
        start = (step < 0) ? length - 1 : length;
    }

    if (stop < 0)
    {
        stop += length;
        if (stop < 0)
        {
            stop = (step < 0) ? -1 : 0;
        }
    }
    else if (stop >= length)
    {
        stop = (step < 0) ? length - 1 : length;
    }

    if (step < 0)
    {
        if (stop < start)
        {
            return (start - stop - 1) / (-step) + 1;
        }
    }
    else if (start < stop)
    {
        return (stop - start - 1) / step + 1;
    }
    return 0;
}

} // namespace detail

template <std::random_access_iterator Iterator>
class SliceIterator
{
public:
    using iterator_concept = std::random_access_iterator_tag;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::iter_value_t<Iterator>;
    using difference_type = std::ptrdiff_t;
    using reference = std::iter_reference_t<Iterator>;
    using pointer = std::add_pointer_t<std::remove_reference_t<reference>>;

    constexpr SliceIterator() = default;

    constexpr SliceIterator(Iterator first, difference_type index, difference_type step) noexcept :
        m_first(std::move(first)),
        m_index(index),
        m_step(step)
    {
        assert(step != 0);
    }

    [[nodiscard]] constexpr reference operator*() const noexcept
    {
        return m_first[m_index * m_step];
    }

    [[nodiscard]] constexpr pointer operator->() const noexcept
        requires std::is_lvalue_reference_v<reference>
    {
        return std::addressof(m_first[m_index * m_step]);
    }

    [[nodiscard]] constexpr reference operator[](difference_type n) const noexcept
    {
        return m_first[(m_index + n) * m_step];
    }

    constexpr SliceIterator& operator++() noexcept
    {
        ++m_index;
        return *this;
    }

    constexpr SliceIterator operator++(int) noexcept
    {
        SliceIterator tmp = *this;
        ++*this;
        return tmp;
    }

    constexpr SliceIterator& operator--() noexcept
    {
        --m_index;
        return *this;
    }

    constexpr SliceIterator operator--(int) noexcept
    {
        SliceIterator tmp = *this;
        --*this;
        return tmp;
    }

    constexpr SliceIterator& operator+=(difference_type n) noexcept
    {
        m_index += n;
        return *this;
    }

    constexpr SliceIterator& operator-=(difference_type n) noexcept
    {
        m_index -= n;
        return *this;
    }

    [[nodiscard]] friend constexpr SliceIterator operator+(SliceIterator it,
                                                           difference_type n) noexcept
    {
        it += n;
        return it;
    }

    [[nodiscard]] friend constexpr SliceIterator operator+(difference_type n,
                                                           SliceIterator it) noexcept
    {
        it += n;
        return it;
    }

    [[nodiscard]] friend constexpr SliceIterator operator-(SliceIterator it,
                                                           difference_type n) noexcept
    {
        it -= n;
        return it;
    }

    [[nodiscard]] friend constexpr difference_type operator-(const SliceIterator& a,
                                                             const SliceIterator& b) noexcept
    {
        return a.m_index - b.m_index;
    }

    [[nodiscard]] friend constexpr bool operator==(const SliceIterator& a,
                                                   const SliceIterator& b) noexcept
    {
        return a.m_index == b.m_index;
    }

    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(const SliceIterator& a,
                                                                    const SliceIterator& b) noexcept
    {
        return a.m_index <=> b.m_index;
    }

private:
    Iterator m_first{};
    difference_type m_index = 0;
    difference_type m_step = 1;
}; // class SliceIterator

// Random-access view of every step-th element in [start, stop). Does not own the underlying
// range; that range must outlive the view. Negative start/stop follow Python slice rules.
template <std::random_access_iterator Iterator>
class [[nodiscard]] SliceView : public std::ranges::view_interface<SliceView<Iterator>>
{
public:
    using iterator = SliceIterator<Iterator>;

    constexpr SliceView() = default;

    constexpr SliceView(Iterator first, std::ptrdiff_t count, std::ptrdiff_t step) noexcept :
        m_first(std::move(first)),
        m_count(count),
        m_step(step)
    {
        assert(count >= 0);
        assert(step != 0);
    }

    [[nodiscard]] constexpr iterator begin() const noexcept { return iterator{m_first, 0, m_step}; }

    [[nodiscard]] constexpr iterator end() const noexcept
    {
        return iterator{m_first, m_count, m_step};
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept
    {
        return static_cast<std::size_t>(m_count);
    }

    [[nodiscard]] constexpr std::ptrdiff_t step() const noexcept { return m_step; }

private:
    Iterator m_first{};
    std::ptrdiff_t m_count = 0;
    std::ptrdiff_t m_step = 1;
};

template <SizedRandomAccessRange Range>
    requires std::is_lvalue_reference_v<Range> ||
             std::ranges::borrowed_range<std::remove_cvref_t<Range>>
[[nodiscard]] constexpr SliceView<std::ranges::iterator_t<Range>> Slice(
    Range&& r, std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step = 1) noexcept
{
    assert(step != 0);

    const auto size = std::ranges::size(r);
    assert(size <= static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()));

    std::ptrdiff_t first = start;
    std::ptrdiff_t last = stop;
    const std::ptrdiff_t count =
        detail::NormalizeSlice(static_cast<std::ptrdiff_t>(size), first, last, step);

    auto it = std::ranges::begin(r);
    if (count > 0)
    {
        it += first;
    }
    return SliceView<std::ranges::iterator_t<Range>>{std::move(it), count, step};
}

} // namespace rad

template <std::random_access_iterator Iterator>
inline constexpr bool std::ranges::enable_borrowed_range<rad::SliceView<Iterator>> = true;
