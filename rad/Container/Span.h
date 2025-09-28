#pragma once

#include <rad/Common/Platform.h>

#include <span>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <type_traits>
#include <vector>

#include <ranges>

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
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
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

    // Construct an ArrayRef from a single element.
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

    template<std::ranges::contiguous_range R>
    constexpr ArrayRef(const R& r) :
        m_data(std::ranges::data(r)),
        m_count(std::ranges::size(r))
    {
    }

    template <size_t N>
    constexpr ArrayRef(const T(&arr)[N]) noexcept :
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
    }

    template <typename Element = T, typename std::enable_if_t<std::is_const_v<Element>, int> = 0>
    ArrayRef(std::initializer_list<typename std::remove_const_t<T>> const& list) noexcept :
        m_data(list.begin()),
        m_count(list.size())
    {
    }

#if __GNUC__ >= 9
#pragma GCC diagnostic pop
#endif

    /// Disallow accidental assignment from a temporary.
    ///
    /// The declaration here is extra complicated so that "arrayRef = {}"
    /// continues to select the move assignment operator.
    template <typename U>
    std::enable_if_t<std::is_same<U, T>::value, ArrayRef<T>>&
        operator=(U&& temp) = delete;

    /// Disallow accidental assignment from a temporary.
    ///
    /// The declaration here is extra complicated so that "arrayRef = {}"
    /// continues to select the move assignment operator.
    template <typename U>
    std::enable_if_t<std::is_same<U, T>::value, ArrayRef<T>>&
        operator=(std::initializer_list<U>) = delete;

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
        assert(index < m_count && "Invalid index!");
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

    /// consume_front() - Returns the first element and drops it from ArrayRef.
    const T& consume_front() {
        const T& ret = front();
        *this = drop_front();
        return ret;
    }

    /// consume_back() - Returns the last element and drops it from ArrayRef.
    const T& consume_back() {
        const T& ret = back();
        *this = drop_back();
        return ret;
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
    void drop_front_in_place(size_t n = 1)
    {
        assert(size() >= n && "Dropping more elements than exist");
        m_data = m_data + n;
        m_count = m_count - n;
    }

    /// Drop the last \p n elements of the array.
    void drop_back_in_place(size_t n = 1)
    {
        assert(size() >= n && "Dropping more elements than exist");
        m_count = m_count - n;
    }

    /// Drop the first \p n elements of the array.
    ArrayRef<T> drop_front(size_t n = 1) const
    {
        assert(size() >= n && "Dropping more elements than exist");
        return slice(n, size() - n);
    }

    /// Drop the last \p n elements of the array.
    ArrayRef<T> drop_back(size_t n = 1) const
    {
        assert(size() >= n && "Dropping more elements than exist");
        return slice(0, size() - n);
    }

    /// Return a copy of *this with the first N elements satisfying the
    /// given predicate removed.
    template <class Predicate> ArrayRef<T> drop_while(Predicate pred) const {
        return ArrayRef<T>(std::ranges::find_if_not(*this, pred), end());
    }

    /// Return a copy of *this with the first N elements not satisfying
    /// the given predicate removed.
    template <class Predicate> ArrayRef<T> drop_until(Predicate pred) const {
        return ArrayRef<T>(std::ranges::find_if(*this, pred), end());
    }

    /// Return a copy of *this with only the first \p N elements.
    ArrayRef<T> take_front(size_t n = 1) const {
        if (n >= size())
        {
            return *this;
        }
        return drop_back(size() - n);
    }

    /// Return a copy of *this with only the last \p N elements.
    ArrayRef<T> take_back(size_t n = 1) const {
        if (n >= size())
        {
            return *this;
        }
        return drop_front(size() - n);
    }

    /// Return the first N elements of this Array that satisfy the given predicate.
    template <class Predicate> ArrayRef<T> take_while(Predicate pred) const {
        return ArrayRef<T>(begin(), std::ranges::find_if_not(*this, pred));
    }

    /// Return the first N elements of this Array that don't satisfy the given predicate.
    template <class Predicate> ArrayRef<T> take_until(Predicate pred) const {
        return ArrayRef<T>(begin(), std::ranges::find_if(*this, pred));
    }

    template <typename Allocator = std::allocator<T>>
    std::vector<T, Allocator> copy() const
    {
        return std::vector<T, Allocator>(m_data, m_data + m_count);
    }

    template <typename Allocator = std::allocator<T>>
    operator std::vector<T, Allocator>() const
    {
        return copy<Allocator>();
    }

}; // class ArrayRef

} // namespace rad
