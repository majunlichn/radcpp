#pragma once

#include <rad/Common/TypeTraits.h>
#include <rad/Common/Utility.h>

#include <iterator>
#include <tuple>
#include <boost/iterator/zip_iterator.hpp>

namespace rad
{

template< typename Iterator >
using iterator_value_t = typename std::iterator_traits< Iterator >::value_type;

template< typename Iterator >
using iterator_reference_t = typename std::iterator_traits< Iterator >::reference;

template <typename... Ptrs>
    requires(std::is_pointer_v<Ptrs> && ...)
class ZipPointer
{
public:
    template<typename T>
    using element_ref_t = std::add_lvalue_reference_t<std::remove_pointer_t<T>>;

    using iterator_category = std::random_access_iterator_tag;
    using reference = std::tuple<typename element_ref_t<Ptrs>...>;
    using value_type = reference;
    using difference_type = std::ptrdiff_t;

    std::tuple<Ptrs...> m_ptrs;

    ZipPointer() = default;
    explicit ZipPointer(Ptrs... ptrs)
        : m_ptrs(ptrs...)
    {
    }

    template<size_t Index>
    auto& get()
    {
        return std::get<Index>(m_ptrs);
    }

    template<size_t Index>
    const auto& get() const
    {
        return std::get<Index>(m_ptrs);
    }

    value_type operator*() const
    {
        return std::apply([](auto... ptrs) {
            return std::tie(*ptrs...);
            }, m_ptrs);
    }

    ZipPointer& operator++()
    {
        std::apply([](auto&... ptrs) {
            (++ptrs, ...);
            }, m_ptrs);
        return *this;
    }

    ZipPointer operator++(int) {
        ZipPointer tmp = *this;
        ++(*this);
        return tmp;
    }

    ZipPointer& operator--() {
        std::apply([](auto&... ptrs) {
            (--ptrs, ...);
            }, m_ptrs);
        return *this;
    }

    ZipPointer operator--(int) {
        ZipPointer tmp = *this;
        --(*this);
        return tmp;
    }

    ZipPointer& operator+=(difference_type n) {
        std::apply([n](auto&... ptrs) {
            ((ptrs += n), ...);
            }, m_ptrs);
        return *this;
    }

    ZipPointer& operator-=(difference_type n) {
        return *this += (-n);
    }

    ZipPointer operator+(difference_type n) const {
        ZipPointer tmp = *this;
        tmp += n;
        return tmp;
    }

    ZipPointer operator-(difference_type n) const {
        ZipPointer tmp = *this;
        tmp -= n;
        return tmp;
    }

    difference_type operator-(const ZipPointer& other) const {
        return std::get<0>(m_ptrs) - std::get<0>(other.m_ptrs);
    }

    reference operator[](difference_type n) const {
        return *(*this + n);
    }

    bool operator==(const ZipPointer& other) const {
        return m_ptrs == other.m_ptrs;
    }

    bool operator!=(const ZipPointer& other) const {
        return !(*this == other);
    }

    bool operator<(const ZipPointer& other) const {
        return std::get<0>(m_ptrs) < std::get<0>(other.m_ptrs);
    }

    bool operator>(const ZipPointer& other) const {
        return other < *this;
    }

    bool operator<=(const ZipPointer& other) const {
        return !(*this > other);
    }

    bool operator>=(const ZipPointer& other) const {
        return !(*this < other);
    }

}; // class ZipPointer

template<typename... Ptrs>
ZipPointer<Ptrs...> operator+(
    typename ZipPointer<Ptrs...>::difference_type n,
    const ZipPointer<Ptrs...>& p) {
    return p + n;
}

template<typename... Ptrs>
ZipPointer<Ptrs...> operator-(
    typename ZipPointer<Ptrs...>::difference_type n,
    const ZipPointer<Ptrs...>& p) {
    return p - n;
}

template<typename... Ptrs>
auto MakeZipPointer(Ptrs... ptrs) {
    return ZipPointer<Ptrs...>(ptrs...);
}

static_assert(std::random_access_iterator<ZipPointer<const float*, int*>>);

} // namespace rad
