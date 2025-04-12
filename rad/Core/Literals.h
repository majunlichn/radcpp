#pragma once

#include <rad/Core/Platform.h>
#include <rad/Core/Integer.h>

namespace rad
{

constexpr uint64_t operator""_KiB(uint64_t x)
{
    return x * 1024;
}

constexpr uint64_t operator""_MiB(uint64_t x)
{
    return x * 1024_KiB;
}

constexpr uint64_t operator""_GiB(uint64_t x)
{
    return x * 1024_MiB;
}

constexpr uint64_t operator""_TiB(uint64_t x)
{
    return x * 1024_GiB;
}

constexpr uint64_t operator""_PiB(uint64_t x)
{
    return x * 1024_TiB;
}

} // namespace rad
