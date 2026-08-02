#include <rad/Core/Platform.h>

#include <gtest/gtest.h>

#include <iostream>
#include <format>

namespace
{

constexpr const char* GetArchitectureName()
{
#if defined(RAD_ARCH_X86_32)
    return "x86";
#elif defined(RAD_ARCH_X86_64)
    return "x86-64";
#elif defined(RAD_ARCH_ARM)
    return "arm";
#elif defined(RAD_ARCH_AARCH64)
    return "aarch64";
#elif defined(RAD_ARCH_MIPS32)
    return "mips32";
#elif defined(RAD_ARCH_MIPS64)
    return "mips64";
#elif defined(RAD_ARCH_PPC)
    return "ppc";
#elif defined(RAD_ARCH_S390X)
    return "s390x";
#elif defined(RAD_ARCH_RISCV32)
    return "riscv32";
#elif defined(RAD_ARCH_RISCV64)
    return "riscv64";
#elif defined(RAD_ARCH_RISCV128)
    return "riscv128";
#elif defined(RAD_ARCH_RISCV)
    return "riscv";
#elif defined(RAD_ARCH_LOONGARCH)
    return "loongarch";
#elif defined(RAD_ARCH_VM)
    return "VM";
#else
    return "Unknown";
#endif
}

constexpr const char* GetOperatingSystemName()
{
#if defined(RAD_OS_WINDOWS)
    return "Windows";
#elif defined(RAD_OS_MACOS)
    return "macOS";
#elif defined(RAD_OS_IPHONE)
    return "iPhone";
#elif defined(RAD_OS_ANDROID)
    return "Android";
#elif defined(RAD_OS_LINUX)
    return "Linux";
#elif defined(RAD_OS_FREEBSD)
    return "FreeBSD";
#elif defined(RAD_OS_OPENBSD)
    return "OpenBSD";
#else
    return "Unknown";
#endif
}

constexpr const char* GetCompilerName()
{
#if defined(RAD_COMPILER_CLANG)
    return "Clang";
#elif defined(RAD_COMPILER_GCC)
    return "GCC";
#elif defined(RAD_COMPILER_MSVC)
    return "MSVC";
#else
    return "Unknown";
#endif
}

} // namespace

TEST(Core, Platform)
{
    std::cout << std::format("Compiled by {} target {} for {}\n", GetCompilerName(),
                             GetArchitectureName(), GetOperatingSystemName());

#if defined(RAD_ARCH_X86_64)
    EXPECT_TRUE(RAD_COMPILED_X86_SSE);
    EXPECT_TRUE(RAD_COMPILED_X86_SSE2);
#endif
}
