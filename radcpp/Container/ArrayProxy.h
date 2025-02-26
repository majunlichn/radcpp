#pragma once

#include <radcpp/Core/Platform.h>
#include <radcpp/Core/TypeTraits.h>

#include <cassert>
#include <initializer_list>

namespace rad
{

// https://github.com/KhronosGroup/Vulkan-Hpp

template <typename T>
class ArrayProxy
{
public:
    constexpr ArrayProxy() noexcept
        : m_data(nullptr), m_count(0)
    {
    }

    constexpr ArrayProxy(std::nullptr_t) noexcept
        : m_data(nullptr), m_count(0)
    {
    }

    ArrayProxy(T& value) noexcept
        : m_data(&value), m_count(1)
    {
    }

    ArrayProxy(T* first, uint32_t count) noexcept
        : m_data(first), m_count(count)
    {
    }

    template <std::size_t C>
    ArrayProxy(T(&ptr)[C]) noexcept
        : m_data(ptr), m_count(C)
    {
    }

#  if __GNUC__ >= 9
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Winit-list-lifetime"
#  endif

    ArrayProxy(std::initializer_list<T> const& list) noexcept
        : m_data(list.begin()), m_count(static_cast<uint32_t>(list.size()))
    {
    }

    template <typename B = T, typename std::enable_if<std::is_const<B>::value, int>::type = 0>
    ArrayProxy(std::initializer_list<typename std::remove_const<T>::type> const& list) noexcept
        : m_data(list.begin()), m_count(static_cast<uint32_t>(list.size()))
    {
    }

#  if __GNUC__ >= 9
#    pragma GCC diagnostic pop
#  endif

    // Any type with a .data() return type implicitly convertible to T*, and a .size() return type implicitly
    // convertible to size_t. The const version can capture temporaries, with lifetime ending at end of statement.
    template <typename V,
        typename std::enable_if<std::is_convertible<decltype(std::declval<V>().data()), T*>::value&&
        std::is_convertible<decltype(std::declval<V>().size()), std::size_t>::value>::type* = nullptr>
    ArrayProxy(V& v) noexcept
        : m_data(v.data()), m_count(static_cast<uint32_t>(v.size()))
    {
    }

    T* begin() const noexcept
    {
        return m_data;
    }

    T* end() const noexcept
    {
        return m_data + m_count;
    }

    T& front() const noexcept
    {
        assert(m_data && m_count);
        return *m_data;
    }

    T& back() const noexcept
    {
        assert(m_data && m_count);
        return *(m_data + m_count - 1);
    }

    bool empty() const noexcept
    {
        return (m_count == 0);
    }

    uint32_t size() const noexcept
    {
        return m_count;
    }

    uint32_t count() const noexcept
    {
        return m_count;
    }

    T* data() const noexcept
    {
        return m_data;
    }

    T& operator[](size_t index) const {
        assert(index < m_count);
        return m_data[index];
    }

private:
    T* m_data;
    uint32_t  m_count;
}; // class ArrayProxy

template <typename T>
class ArrayProxyNoTemporaries
{
public:
    constexpr ArrayProxyNoTemporaries() noexcept
        : m_data(nullptr), m_count(0)
    {
    }

    constexpr ArrayProxyNoTemporaries(std::nullptr_t) noexcept
        : m_data(nullptr), m_count(0)
    {
    }

    template <typename B = T, typename std::enable_if<std::is_convertible<B, T>::value&& std::is_lvalue_reference<B>::value, int>::type = 0>
    ArrayProxyNoTemporaries(B&& value) noexcept
        : m_data(&value), m_count(1)
    {
    }

    ArrayProxyNoTemporaries(T* first, uint32_t count) noexcept
        : m_data(first), m_count(count)
    {
    }

    template <std::size_t C>
    ArrayProxyNoTemporaries(T(&ptr)[C]) noexcept
        : m_data(ptr), m_count(C)
    {
    }

    template <std::size_t C>
    ArrayProxyNoTemporaries(T(&& ptr)[C]) = delete;

    // Any l-value reference with a .data() return type implicitly convertible to T*, and a .size() return type implicitly convertible to size_t.
    template <typename V,
        typename std::enable_if<!std::is_convertible<decltype(std::declval<V>().begin()), T*>::value&&
        std::is_convertible<decltype(std::declval<V>().data()), T*>::value&&
        std::is_convertible<decltype(std::declval<V>().size()), std::size_t>::value&& std::is_lvalue_reference<V>::value,
        int>::type = 0>
    ArrayProxyNoTemporaries(V&& v) noexcept
        : m_data(v.data()), m_count(static_cast<uint32_t>(v.size()))
    {
    }

    // Any l-value reference with a .begin() return type implicitly convertible to T*, and a .size() return type implicitly convertible to size_t.
    template <typename V,
        typename std::enable_if<std::is_convertible<decltype(std::declval<V>().begin()), T*>::value&&
        std::is_convertible<decltype(std::declval<V>().size()), std::size_t>::value&& std::is_lvalue_reference<V>::value,
        int>::type = 0>
    ArrayProxyNoTemporaries(V&& v) noexcept
        : m_data(v.begin()), m_count(static_cast<uint32_t>(v.size()))
    {
    }

    T* begin() const noexcept
    {
        return m_data;
    }

    T* end() const noexcept
    {
        return m_data + m_count;
    }

    T& front() const noexcept
    {
        assert(m_data && m_count);
        return *m_data;
    }

    T& back() const noexcept
    {
        assert(m_data && m_count);
        return *(m_data + m_count - 1);
    }

    bool empty() const noexcept
    {
        return (m_count == 0);
    }

    uint32_t size() const noexcept
    {
        return m_count;
    }

    uint32_t count() const noexcept
    {
        return m_count;
    }

    T* data() const noexcept
    {
        return m_data;
    }

    T& operator[](size_t index) const {
        assert(index < m_count);
        return m_data[index];
    }

private:
    T* m_data;
    uint32_t m_count;
}; // class ArrayProxyNoTemporaries

template <typename T>
class StridedArrayProxy
{
public:
    using Byte = typename std::conditional_t<std::is_const_v<T>, const uint8_t, uint8_t>;

    struct Iterator
    {
        Iterator(Byte* ptr, uint32_t stride) : m_ptr(ptr), m_stride(stride) {}
        ~Iterator() {}

        operator T* () { return reinterpret_cast<T*>(m_ptr); }
        Iterator& operator++() { m_ptr += m_stride; return *this; }
        Iterator operator++(int) { Iterator old = *this; operator++(); return old; }
        Iterator& operator--() { m_ptr -= m_stride; return *this; }
        Iterator operator--(int) { Iterator old = *this; operator--(); return old; }

        Byte* m_ptr;
        uint32_t m_stride;
    };

    StridedArrayProxy(T* first, uint32_t count, uint32_t stride) noexcept
        : m_data(reinterpret_cast<Byte*>(first)), m_count(count), m_stride(stride)
    {
    }

    Iterator begin() const noexcept
    {
        return Iterator(m_data, m_stride);
    }

    Iterator end() const noexcept
    {
        return Iterator(m_data + size() * m_stride, m_stride);
    }

    T& front() const noexcept
    {
        assert(m_data && m_count && m_stride);
        return *begin();
    }

    T& back() const noexcept
    {
        assert(m_data && m_count && m_stride);
        return *end();
    }

    bool empty() const noexcept
    {
        return (m_count == 0);
    }

    uint32_t size() const noexcept
    {
        return m_count;
    }

    uint32_t count() const noexcept
    {
        return m_count;
    }

    Byte* data() const
    {
        return m_data;
    }

    T& operator[](size_t index) const {
        assert(index < m_count);
        return *reinterpret_cast<T*>(m_data + index * m_stride);
    }

private:
    Byte* m_data;
    uint32_t m_count;
    uint32_t m_stride;

}; // class StridedArrayProxy

} // namespace rad
