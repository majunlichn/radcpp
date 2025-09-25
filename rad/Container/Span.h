#pragma once

#include <rad/Common/Platform.h>

#include <span>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <ranges>
#include <type_traits>
#include <vector>

namespace rad
{

template<typename T>
using Span = std::span<T>;

// Const reference to an array, more flexible and easy to use,, inspired by:
// https://github.com/llvm-mirror/llvm/blob/master/include/llvm/ADT/ArrayRef.h
// https://github.com/KhronosGroup/Vulkan-Hpp/blob/main/vulkan/vulkan.hpp
template <typename T>
class [[nodiscard]] ArrayRef
{
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using iterator = const_pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

private:
    const T* m_data = nullptr;
    size_type m_count = 0;

public:
    constexpr ArrayRef() noexcept :
        m_data(nullptr),
        m_count(0)
    {
    }

    constexpr ArrayRef(std::nullptr_t) noexcept :
        m_data(nullptr),
        m_count(0)
    {
    }

    ArrayRef(const T& value) noexcept :
        m_data(&value),
        m_count(1)
    {
    }

    constexpr ArrayRef(const T* ptr, size_t count) noexcept :
        m_data(ptr),
        m_count(count)
    {
    }

    constexpr ArrayRef(const T* first, const T* last) noexcept :
        m_data(first),
        m_count(last - first)
    {
        assert(first <= last);
    }

    template <size_t N>
    ArrayRef(const T(&arr)[N]) noexcept :
        m_data(arr),
        m_count(N)
    {
    }

#if __GNUC__ >= 9
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-list-lifetime"
#endif

    ArrayRef(std::initializer_list<T> const& list) noexcept :
        m_data(list.begin()),
        m_count(list.size())
    {
        if (m_count == 0)
        {
            m_data = 0;
        }
    }

    template <typename Element = T, typename std::enable_if_t<std::is_const_v<Element>, int> = 0>
    ArrayRef(std::initializer_list<typename std::remove_const_t<T>> const& list) noexcept :
        m_data(list.begin()),
        m_count(list.size())
    {
        if (m_count == 0)
        {
            m_data = 0;
        }
    }

#if __GNUC__ >= 9
#pragma GCC diagnostic pop
#endif

    template<std::ranges::contiguous_range R>
    ArrayRef(const R& r) :
        m_data(std::ranges::data(r)),
        m_count(std::ranges::size(r))
    {
        if (m_count == 0)
        {
            m_data = 0;
        }
    }

    const T* data() const noexcept
    {
        return m_data;
    }

    size_t size() const noexcept
    {
        return m_count;
    }

    uint32_t size32() const noexcept
    {
        return static_cast<uint32_t>(m_count);
    }

    bool empty() const noexcept
    {
        return (m_count == 0);
    }

    iterator begin() const { return m_data; }
    iterator end() const { return m_data + m_count; }

    reverse_iterator rbegin() const { return reverse_iterator(end()); }
    reverse_iterator rend() const { return reverse_iterator(begin()); }

    const T& operator[](size_t index) const
    {
        assert(index < m_count);
        return m_data[index];
    }

    const T& front() noexcept
    {
        assert(m_data && (m_count > 0));
        return *m_data;
    }

    const T& back() noexcept
    {
        assert(m_data && (m_count > 0));
        return *(m_data + m_count - 1);
    }

    /// equals - Check for element-wise equality.
    bool equals(ArrayRef rhs) const
    {
        if (m_count != rhs.m_count)
        {
            return false;
        }
        return std::equal(begin(), end(), rhs.begin());
    }

    /// slice(n, m) - Chop off the first n elements of the array, and keep m
    /// elements in the array.
    ArrayRef<T> slice(size_t n, size_t m) const
    {
        assert(n + m <= size());
        return ArrayRef<T>(data() + n, m);
    }

    /// slice(n) - Chop off the first n elements of the array.
    ArrayRef<T> slice(size_t n) const { return slice(n, size() - n); }

    /// Drop the first \p n elements of the array.
    void drop_front(size_t n = 1)
    {
        assert(size() >= n && "Dropping more elements than exist");
        m_data = m_data + n;
        m_count = m_count - n;
    }

    /// Drop the last \p n elements of the array.
    void drop_back(size_t n = 1)
    {
        assert(size() >= n && "Dropping more elements than exist");
        m_count = m_count - n;
    }

    operator std::vector<T>() const
    {
        return std::vector<T>(m_data, m_data + m_count);
    }

}; // class ArrayRef

} // namespace rad
