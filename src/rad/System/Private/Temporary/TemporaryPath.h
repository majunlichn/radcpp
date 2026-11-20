#pragma once

#include <rad/Core/String.h>
#include <rad/System/OS.h>

#include <stdexcept>
#include <utility>

namespace rad::detail
{

[[nodiscard]] inline Result<FilePath> ResolveTemporaryParent(const FilePath& parent) noexcept
{
    if (parent.native().find(FilePath::value_type{}) != FilePath::string_type::npos)
    {
        return Failure(std::errc::invalid_argument);
    }

    try
    {
        auto path = parent.empty() ? os::temp_directory_path() : parent;
        if (!path.is_absolute())
        {
            std::error_code error;
            path = std::filesystem::absolute(path, error);
            if (error)
            {
                return Failure(error);
            }
        }
        return Success(std::move(path));
    }
    catch (const std::system_error& error)
    {
        return Failure(error.code());
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

[[nodiscard]] constexpr std::string MakeTemporaryFilename(std::string hex,
                                                          FilePath::value_type excludedDigit)
{
    const char first = hex.front();
    if (excludedDigit == first ||
        (first >= 'a' && first <= 'f' && excludedDigit == first - 'a' + 'A'))
    {
        hex.front() = first == '0' ? '1' : '0';
    }
    return "." + std::move(hex) + ".tmp";
}

static_assert(MakeTemporaryFilename("abcdef0123456789abcdef0123456789", 'A') ==
              ".0bcdef0123456789abcdef0123456789.tmp");
static_assert(MakeTemporaryFilename("0123456789abcdef0123456789abcdef", '0') ==
              ".1123456789abcdef0123456789abcdef.tmp");

template <typename Create>
[[nodiscard]] auto CreateUniqueTemporary(const FilePath& parent, Create create,
                                         FilePath::value_type excludedDigit = {}) noexcept
    -> decltype(create(FilePath{}))
{
    try
    {
        for (int attempt = 0; attempt < 8; ++attempt)
        {
            auto path = parent / MakeTemporaryFilename(ToHexString(os::urandom(16)), excludedDigit);
            auto created = create(std::move(path));
            if (created || created.error() != std::errc::file_exists)
            {
                return created;
            }
        }
        return Failure(std::errc::file_exists);
    }
    catch (const std::system_error& error)
    {
        return Failure(error.code());
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

} // namespace rad::detail
