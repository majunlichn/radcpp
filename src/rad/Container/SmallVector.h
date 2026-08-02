#pragma once

#include <cstddef>

#include <boost/container/small_vector.hpp>

namespace rad
{

// SmallVector is a vector-like container optimized for the case when it contains few elements.
// https://www.boost.org/doc/libs/latest/doc/html/container/non_standard_containers.html#container.non_standard_containers.small_vector
// https://llvm.org/docs/ProgrammersManual.html#llvm-adt-smallvector-h
template <class T, std::size_t N>
using SmallVector = boost::container::small_vector<T, N>;

} // namespace rad
