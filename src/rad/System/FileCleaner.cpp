#include <rad/System/FileCleaner.h>

#include <rad/Core/Platform.h>

#include <iterator>
#include <string_view>
#include <system_error>
#include <utility>

#if defined(RAD_OS_WINDOWS)
#include <windows.h>
#include <pathcch.h>
#else
#include <cerrno>
#include <unistd.h>
#endif

namespace rad
{
namespace
{

[[nodiscard]] FilePath ResolvePath(const FilePath& path)
{
    if (path.empty() || path.native().find(FilePath::value_type{}) != FilePath::string_type::npos)
    {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument), "FileCleaner");
    }
#if defined(RAD_OS_WINDOWS)
    // Normalize ordinary Windows paths before checking for drive and share roots.
    auto absolute = std::filesystem::absolute(path);
#else
    auto absolute = path.is_absolute() ? path : std::filesystem::absolute(path);
#endif
    if (absolute.has_relative_path() && !absolute.has_filename())
    {
        absolute = absolute.parent_path();
    }
    return absolute;
}

[[nodiscard]] bool RemoveFileOrLink(const FilePath& path) noexcept
{
#if defined(RAD_OS_WINDOWS)
    if (DeleteFileW(path.c_str()))
    {
        return true;
    }
    const auto error = GetLastError();
    return error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND;
#else
    return ::unlink(path.c_str()) == 0 || errno == ENOENT;
#endif
}

} // namespace

FileCleaner::~FileCleaner() noexcept
{
    try
    {
        static_cast<void>(Clean());
    }
    catch (const std::system_error&)
    {
        // Best effort if acquiring the lock fails.
    }
}

void FileCleaner::Register(const FilePath& path)
{
    const auto name = path.has_filename() ? path.filename() : path.parent_path().filename();
    if (name == "." || name == "..")
    {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument), "FileCleaner");
    }

    auto absolute = ResolvePath(path);
    if (!absolute.has_filename())
    {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument), "FileCleaner");
    }
#if defined(RAD_OS_WINDOWS)
    // filesystem::path does not recognize share names as part of Windows roots.
    const wchar_t* afterRoot = nullptr;
    if (FAILED(PathCchSkipRoot(absolute.c_str(), &afterRoot)) ||
        std::wstring_view(afterRoot).find_first_not_of(L"\\/") == std::wstring_view::npos)
    {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument), "FileCleaner");
    }
#endif

    std::lock_guard lock(m_mutex);
    m_paths.insert(std::move(absolute));
}

void FileCleaner::Unregister(const FilePath& path)
{
    const auto absolute = ResolvePath(path);
    std::lock_guard lock(m_mutex);
    m_paths.erase(absolute);
}

bool FileCleaner::IsRegistered(const FilePath& path) const
{
    const auto absolute = ResolvePath(path);
    std::lock_guard lock(m_mutex);
    return m_paths.contains(absolute);
}

bool FileCleaner::Clean()
{
    std::lock_guard lock(m_mutex);
    for (auto entry = m_paths.begin(); entry != m_paths.end();)
    {
        entry = RemoveFileOrLink(*entry) ? m_paths.erase(entry) : std::next(entry);
    }
    return m_paths.empty();
}

} // namespace rad
