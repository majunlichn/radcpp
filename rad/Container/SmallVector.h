#pragma once

#include <rad/Core/Platform.h>
#include <boost/container/small_vector.hpp>

namespace rad
{

// SmallVector contains some preallocated elements in-place,
// which can avoid dynamic storage allocation when the actual number of elements
// is below that preallocated threshold.
template<class T, std::size_t N>
using SmallVector = boost::container::small_vector<T, N>;

}
