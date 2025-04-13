#include <rad/System/OS.h>

#if defined(RAD_OS_WINDOWS)
#include <Windows.h>
#include <process.h>
#include <VersionHelpers.h>
#else
#include <unistd.h>
#include <limits.h>
#include <pwd.h>
#endif

#include <boost/process.hpp>

namespace rad
{

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
#elif defined(RAD_OS_LINUX)
    std::string username = getenv("LOGNAME");
    if (!username.empty())
    {
        return username;
    }
    username = getenv("USERNAME");
    if (!username.empty())
    {
        return username;
    }
    const passwd* pw = getpwuid(geteuid());
    if (pw != nullptr)
    {
        return pw->pw_name;
    }
    username.resize(LOGIN_NAME_MAX);
    int err = getlogin_r(username.data(), username.size());
    if (err == 0)
    {
        return username;
    }
    return {};
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

std::vector<std::string> ExecuteAndReadLines(const std::string& command)
{
    using namespace boost;
    process::ipstream stream; // reading pipe-stream
    process::child child(command, process::std_out > stream);

    std::vector<std::string> lines;
    std::string line;
    while (child.running() && std::getline(stream, line) && !line.empty())
    {
        lines.push_back(line);
    }
    child.wait();
    return lines;
}

} // namespace rad
