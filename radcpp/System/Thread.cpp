#include <radcpp/System/Thread.h>
#include <radcpp/Core/String.h>
#include <array>

#if defined(RAD_OS_WINDOWS)
#include <Windows.h>
#endif

#if defined(__GLIBC__) && __GLIBC_PREREQ(2, 12) && !defined(__APPLE__) && \
    !defined(__ANDROID__)
#define HAS_PTHREAD_SETNAME_NP
#endif

#if defined(HAS_PTHREAD_SETNAME_NP)
#include <pthread.h>
#define MAX_THREAD_NAME_LEN size_t(15)
#endif

namespace rad
{

bool SetThreadName(std::string name)
{
#if defined(_WIN32)
    std::wstring nameWide = rad::ToWideString(name);
    HRESULT hr = ::SetThreadDescription(::GetCurrentThread(), nameWide.c_str());
    return SUCCEEDED(hr);
#elif defined(HAS_PTHREAD_SETNAME_NP)
    name.resize(std::min<size_t>(name.size(), MAX_THREAD_NAME_LEN);
    int err = pthread_setname_np(pthread_self(), name.c_str());
    return (err == 0);
#else
    return false;
#endif
}

std::string GetThreadName()
{
#if defined(_WIN32)
    PWSTR ppszThreadDescription = NULL;
    HRESULT hr = ::GetThreadDescription(::GetCurrentThread(), &ppszThreadDescription);
    if (SUCCEEDED(hr))
    {
        std::wstring nameWide = ppszThreadDescription;
        LocalFree(ppszThreadDescription);
        ppszThreadDescription = NULL;
        return rad::ToString(nameWide);
    }
    else
    {
        return {};
    }
#elif defined(HAS_PTHREAD_SETNAME_NP)
    std::array<char, MAX_THREAD_NAME_LEN + 1> name{};
    pthread_getname_np(pthread_self(), name.data(), name.size());
    return name.data();
#else
    return {};
#endif
}

} // namespace rad
