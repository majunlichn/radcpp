#include <rad/System/OS.h>
#include <rad/IO/FileSystem.h>
#include <rad/IO/Logging.h>
#if defined(RAD_OS_WINDOWS)
#include <Windows.h>
#include <process.h>
#include <VersionHelpers.h>
#else
#include <unistd.h>
#endif

namespace rad
{

EnvMap getenvs()
{
    EnvMap envs;
    for (char** iter = environ; *iter != nullptr; iter++)
    {
        std::string_view env = std::string_view(*iter);
        size_t sep = env.find_first_of('=');
        std::string_view key = env.substr(0, sep);
        std::string_view value = env.substr(sep + 1);
        envs[std::string(key)] = value;
    }
    return envs;
}

std::string getenv(std::string_view name)
{
#if defined(RAD_OS_WINDOWS)
    char* buffer = nullptr;
    size_t size = 0;
    errno_t err = _dupenv_s(&buffer, &size, name.data());
    if ((err == 0) && (buffer != nullptr))
    {
        std::string value(buffer);
        free(buffer);
        return value;
    }
    else
    {
        return std::string();
    }
#else
    const char* value = std::getenv(name.data());
    if (value)
    {
        return std::string(value);
    }
    else
    {
        return std::string();
    }
#endif
}

bool putenv(std::string_view key, std::string_view value)
{
#if defined(RAD_OS_WINDOWS)
    errno_t err = _putenv_s(key.data(), value.data());
    return (err == 0);
#else
    int err = ::setenv(key.data(), value.data(), 1);
    return (err == 0);
#endif
}

void chdir(std::string_view p)
{
    std::filesystem::current_path((const char8_t*)p.data());
}

std::string getcwd()
{
    return (const char*)std::filesystem::current_path().u8string().c_str();
}

std::vector<std::string> get_exec_path()
{
    std::vector<std::string> paths;
    for (char** iter = environ; *iter != nullptr; iter++)
    {
        std::string_view env = std::string_view(*iter);
        if (env.starts_with("Path") || env.starts_with("PATH"))
        {
            std::string_view value = env.substr(env.find_first_of('=') + 1);
#if defined(RAD_OS_WINDOWS)
            return StrSplit(value, ";", true);
#else
            return StrSplit(value, ":", true);
#endif
        }
    }
    return {};
}

std::string getlogin()
{
#if defined(RAD_OS_WINDOWS)
    std::wstring buffer(1024, 0);
    unsigned long count = static_cast<unsigned long>(buffer.size());
    if (::GetUserNameW(buffer.data(), &count))
    {
        return StrFromWide(buffer);
    }
    else
    {
        return {};
    }
#endif
}

int getpid()
{
#if defined(RAD_OS_WINDOWS)
    return ::_getpid();
#else
    return ::getpid();
#endif
}

int system(std::string_view command)
{
    return std::system(command.data());
}

} // namespace rad
