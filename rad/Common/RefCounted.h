#pragma once

#include <rad/Common/Platform.h>

#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>

namespace rad
{

template<typename T>
using RefCounted = boost::intrusive_ref_counter<T, boost::thread_unsafe_counter>;

template<typename T>
using Ref = boost::intrusive_ptr<T>;

template <class T, class... Types>
[[nodiscard]] Ref<T> MakeRefCounted(Types&&... args)
{
    return RAD_NEW T(std::forward<Types>(args)...);
}

} // namespace rad

#define RAD_MAKE_REFCOUNTED(T, ...) rad::Ref<T>(RAD_NEW T(__VA_ARGS__))
