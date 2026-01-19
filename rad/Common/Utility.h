#pragma once

#include <rad/Common/Platform.h>
#include <utility>

namespace rad
{

// The fallback of std::unreachable():
// https://en.cppreference.com/w/cpp/utility/unreachable.html
[[noreturn]] inline void unreachable()
{
    // Uses compiler specific extensions if possible.
    // Even if no extension is used, undefined behavior is still raised by
    // an empty function body and the noreturn attribute.
#if defined(_MSC_VER) && !defined(__clang__) // MSVC
    __assume(false);
#else // GCC, Clang
    __builtin_unreachable();
#endif
}

} // namespace rad
