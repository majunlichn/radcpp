#include <rad/System/Thread.h>

#include <rad/Core/Platform.h>

#if defined(RAD_OS_WINDOWS)
#include <rad/Core/Unicode.h>

#include <Windows.h>
#elif defined(RAD_OS_LINUX)
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(RAD_OS_ANDROID)
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(RAD_OS_MACOS) || defined(RAD_OS_IPHONE)
#include <pthread.h>
#elif defined(RAD_OS_FREEBSD)
#include <pthread_np.h>
#elif defined(RAD_OS_OPENBSD)
#include <pthread_np.h>
#include <unistd.h>
#else
#error "Thread utilities are not implemented for this platform"
#endif

namespace rad
{

bool SetThreadName(std::string_view name)
{
#if defined(RAD_OS_WINDOWS)
    try
    {
        const std::wstring wideName = Utf8ToWide(name);
        return SUCCEEDED(::SetThreadDescription(::GetCurrentThread(), wideName.c_str()));
    }
    catch (...)
    {
        return false;
    }
#elif defined(RAD_OS_LINUX)
    return ::pthread_setname_np(::pthread_self(), name.empty() ? "" : name.data()) == 0;
#elif defined(RAD_OS_ANDROID)
    if (name.size() >= 16)
    {
        return false;
    }
    return ::prctl(PR_SET_NAME, name.empty() ? "" : name.data(), 0, 0, 0) == 0;
#elif defined(RAD_OS_MACOS) || defined(RAD_OS_IPHONE)
    return ::pthread_setname_np(name.empty() ? "" : name.data()) == 0;
#elif defined(RAD_OS_FREEBSD) || defined(RAD_OS_OPENBSD)
    ::pthread_set_name_np(::pthread_self(), name.empty() ? "" : name.data());
    return true;
#endif
}

std::string GetThreadName()
{
#if defined(RAD_OS_WINDOWS)
    PWSTR wideName = nullptr;
    if (FAILED(::GetThreadDescription(::GetCurrentThread(), &wideName)))
    {
        return {};
    }

    try
    {
        std::string name = WideToUtf8(wideName);
        ::LocalFree(wideName);
        return name;
    }
    catch (...)
    {
        ::LocalFree(wideName);
        return {};
    }
#elif defined(RAD_OS_LINUX)
    char name[16] = {};
    if (::pthread_getname_np(::pthread_self(), name, sizeof(name)) != 0)
    {
        return {};
    }
    return name;
#elif defined(RAD_OS_ANDROID)
    char name[16] = {};
    if (::prctl(PR_GET_NAME, name, 0, 0, 0) != 0)
    {
        return {};
    }
    return name;
#elif defined(RAD_OS_MACOS) || defined(RAD_OS_IPHONE)
    char name[64] = {};
    if (::pthread_getname_np(::pthread_self(), name, sizeof(name)) != 0)
    {
        return {};
    }
    return name;
#elif defined(RAD_OS_FREEBSD) || defined(RAD_OS_OPENBSD)
    char name[64] = {};
    ::pthread_get_name_np(::pthread_self(), name, sizeof(name));
    return name;
#endif
}

std::uint64_t GetCurrentThreadId()
{
#if defined(RAD_OS_WINDOWS)
    return static_cast<std::uint64_t>(::GetCurrentThreadId());
#elif defined(RAD_OS_LINUX) || defined(RAD_OS_ANDROID)
    return static_cast<std::uint64_t>(::syscall(SYS_gettid));
#elif defined(RAD_OS_MACOS) || defined(RAD_OS_IPHONE)
    std::uint64_t threadId = 0;
    ::pthread_threadid_np(nullptr, &threadId);
    return threadId;
#elif defined(RAD_OS_FREEBSD)
    return static_cast<std::uint64_t>(::pthread_getthreadid_np());
#elif defined(RAD_OS_OPENBSD)
    return static_cast<std::uint64_t>(::getthrid());
#endif
}

} // namespace rad
