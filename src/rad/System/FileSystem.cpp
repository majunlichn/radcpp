#include <rad/System/FileSystem.h>

#include <limits>
#include <system_error>

namespace rad
{
namespace
{

[[nodiscard]] bool IsValidPath(const FilePath& path) noexcept
{
    return !path.empty() &&
           path.native().find(FilePath::value_type{}) == FilePath::string_type::npos;
}

[[nodiscard]] std::filesystem::file_status QueryStatus(const FilePath& path,
                                                       std::error_code& ec) noexcept
{
    if (!IsValidPath(path))
    {
        ec = std::make_error_code(std::errc::invalid_argument);
        return {};
    }

    const auto status = std::filesystem::status(path, ec);
    if (status.type() == FileType::not_found)
    {
        ec.clear();
    }
    return status;
}

} // namespace

bool Exists(const FilePath& path) noexcept
{
    std::error_code error;
    return Exists(path, error);
}

bool Exists(const FilePath& path, std::error_code& ec) noexcept
{
    const auto status = QueryStatus(path, ec);
    return !ec && std::filesystem::exists(status);
}

bool IsRegularFile(const FilePath& path) noexcept
{
    std::error_code error;
    return IsRegularFile(path, error);
}

bool IsRegularFile(const FilePath& path, std::error_code& ec) noexcept
{
    const auto status = QueryStatus(path, ec);
    return !ec && std::filesystem::is_regular_file(status);
}

Result<std::uint64_t> GetSize(const FilePath& path) noexcept
{
    if (!IsValidPath(path))
    {
        return Failure(std::errc::invalid_argument);
    }

    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error)
    {
        return Failure(error);
    }
    if (size > std::numeric_limits<std::uint64_t>::max())
    {
        return Failure(std::errc::value_too_large);
    }
    return Success(static_cast<std::uint64_t>(size));
}

Result<FileTime> LastWriteTime(const FilePath& path) noexcept
{
    if (!IsValidPath(path))
    {
        return Failure(std::errc::invalid_argument);
    }

    std::error_code error;
    const auto time = std::filesystem::last_write_time(path, error);
    if (error)
    {
        return Failure(error);
    }
    return Success(time);
}

Result<void> Remove(const FilePath& path) noexcept
{
    if (!IsValidPath(path))
    {
        return Failure(std::errc::invalid_argument);
    }

    std::error_code error;
    if (std::filesystem::remove(path, error))
    {
        return Success();
    }
    return Failure(error ? error : std::make_error_code(std::errc::no_such_file_or_directory));
}

Result<void> Copy(const FilePath& source, const FilePath& destination, bool overwrite) noexcept
{
    if (!IsValidPath(source) || !IsValidPath(destination))
    {
        return Failure(std::errc::invalid_argument);
    }

    const auto options = overwrite ? std::filesystem::copy_options::overwrite_existing
                                   : std::filesystem::copy_options::none;
    std::error_code error;
    if (std::filesystem::copy_file(source, destination, options, error))
    {
        return Success();
    }
    return Failure(error ? error : std::make_error_code(std::errc::io_error));
}

Result<void> Rename(const FilePath& source, const FilePath& destination) noexcept
{
    if (!IsValidPath(source) || !IsValidPath(destination))
    {
        return Failure(std::errc::invalid_argument);
    }
    std::error_code error;
    std::filesystem::rename(source, destination, error);
    if (error)
    {
        return Failure(error);
    }
    return Success();
}

} // namespace rad
