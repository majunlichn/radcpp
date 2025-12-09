#pragma once

#include <rad/Common/Platform.h>

#include <span>


namespace rad
{

template<typename T>
using Span = std::span<T>;

} // namespace rad
