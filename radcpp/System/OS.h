#pragma once

#include <radcpp/Core/Platform.h>
#include <radcpp/Core/String.h>
#include <map>

namespace rad
{

#if defined(RAD_OS_WINDOWS)
using EnvMap = std::map<std::string, std::string, StringLessCaseInsensitive>;
#else
using EnvMap = std::map<std::string, std::string, StringLess>;
#endif
EnvMap getenvs();
std::string getenv(std::string_view name);
bool putenv(std::string_view key, std::string_view value);

void chdir(std::string_view p);
#ifdef getcwd
#undef getcwd
#endif
std::string getcwd();
std::vector<std::string> get_exec_path();

std::string getlogin();

int getpid();

int system(std::string_view command);

namespace path
{

using namespace pystring::os::path;

std::string GetFileName(std::string_view path);
std::string GetFileNameNoExt(std::string_view path);

} // namespace path

} // namespace rad
