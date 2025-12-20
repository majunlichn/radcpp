#include <rad/System/OS.h>
#include <rad/IO/File.h>
#include <rad/IO/Logging.h>

#if defined(RAD_OS_WINDOWS)
#include <Windows.h>
#include <process.h>
#include <VersionHelpers.h>
#else
#include <unistd.h>
#include <limits.h>
#include <pwd.h>
#endif

#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/asio.hpp>

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

// https://github.com/boostorg/process/blob/develop/example/stdio.cpp
std::vector<std::string> ExecuteAndReadLines(std::string_view executable, const std::vector<std::string>& args)
{
    std::vector<std::string> lines;
    try
    {
        boost::filesystem::path path(executable);
        auto env = boost::process::environment::current();
        if (!boost::filesystem::exists(path))
        {
            path = boost::process::environment::find_executable(path, env);
            RAD_LOG(info, "ExecuteAndReadLines({}): found in environment: {}", executable, path.string());
        }
        if (!boost::filesystem::exists(path))
        {
            RAD_LOG(err, "ExecuteAndReadLines({}): executable not found!", executable);
        }
        boost::asio::io_context io;
        boost::process::popen proc(io, path, args, boost::process::process_environment(env));
        std::string buffer;
        auto bufferAdapter = boost::asio::dynamic_buffer(buffer);
        boost::system::error_code ec;
        while (true)
        {
            size_t n = boost::asio::read_until(proc, bufferAdapter, '\n', ec);
            if (n > 0)
            {
                lines.push_back(buffer.substr(0, n));
                buffer.erase(0, n);
            }
            if (ec)
            {
                if ((ec != boost::asio::error::eof) || (ec != boost::asio::error::broken_pipe))
                {
                    RAD_LOG(err, "ExecuteAndReadLines({}): {}", executable, ec.message());
                }
                break;
            }
        }
    }
    catch (std::exception& e)
    {
        RAD_LOG(err, "ExecuteAndReadLines({}) exception: {}", executable, e.what());
    }
    catch (...)
    {
        RAD_LOG(err, "ExecuteAndReadLines({}): unkown exception!", executable);
    }

    return lines;
}

} // namespace rad
