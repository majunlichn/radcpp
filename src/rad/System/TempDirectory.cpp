#include <rad/System/TempDirectory.h>

#include <rad/Core/Platform.h>
#include <rad/System/Private/Temporary/TemporaryPath.h>

#include <stdexcept>
#include <utility>

#if defined(RAD_OS_WINDOWS)
#include <windows.h>
#else
#include <cerrno>
#include <sys/stat.h>
#endif

namespace rad
{

TempDirectory::TempDirectory(FilePath path) noexcept :
    m_path(std::move(path))
{
}

TempDirectory::~TempDirectory() noexcept
{
    static_cast<void>(Clean());
}

TempDirectory::TempDirectory(TempDirectory&& other) noexcept :
    m_path(std::exchange(other.m_path, {}))
{
}

TempDirectory& TempDirectory::operator=(TempDirectory&& other) noexcept
{
    if (this != &other)
    {
        static_cast<void>(Clean());
        m_path = std::exchange(other.m_path, {});
    }
    return *this;
}

Result<TempDirectory> TempDirectory::Create(const FilePath& parent) noexcept
{
    const auto resolved = detail::ResolveTemporaryParent(parent);
    if (!resolved)
    {
        return Failure(resolved.error());
    }
    return detail::CreateUniqueTemporary(
        resolved.value(),
        [](FilePath path) -> Result<TempDirectory>
        {
#if defined(RAD_OS_WINDOWS)
            if (!CreateDirectoryW(path.c_str(), nullptr))
            {
                return Failure(
                    std::error_code(static_cast<int>(GetLastError()), std::system_category()));
            }
#else
            if (::mkdir(path.c_str(), 0700) != 0)
            {
                return Failure(std::error_code(errno, std::generic_category()));
            }
#endif
            return Success(TempDirectory(std::move(path)));
        });
}

const FilePath& TempDirectory::Path() const noexcept
{
    return m_path;
}

FilePath TempDirectory::Release() noexcept
{
    return std::exchange(m_path, {});
}

Result<void> TempDirectory::Clean() noexcept
{
    if (m_path.empty())
    {
        return Success();
    }
    try
    {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
        if (error)
        {
            return Failure(error);
        }
        m_path.clear();
        return Success();
    }
    catch (const std::bad_alloc&)
    {
        return Failure(std::errc::not_enough_memory);
    }
    catch (const std::length_error&)
    {
        return Failure(std::errc::filename_too_long);
    }
}

} // namespace rad
