#pragma once

#include <rad/Core/Platform.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef environ
#undef environ
#endif

namespace rad::os
{

// FilePath uses the platform-native path encoding. On Windows, creating symbolic links may require
// Developer Mode or elevation, and permission values represent the limited Windows filesystem
// permission model.
using FilePath = std::filesystem::path;
using FileTime = std::chrono::system_clock::time_point;

#if defined(RAD_OS_WINDOWS)
inline constexpr std::string_view name = "nt";
inline const FilePath curdir = L".";
inline const FilePath pardir = L"..";
inline constexpr std::string_view sep = "\\";
inline constexpr std::string_view altsep = "/";
inline constexpr std::string_view extsep = ".";
inline constexpr std::string_view pathsep = ";";
inline constexpr std::string_view defpath = ".;C:\\bin";
inline constexpr std::string_view linesep = "\r\n";
inline const FilePath devnull = L"nul";
#else
inline constexpr std::string_view name = "posix";
inline const FilePath curdir = ".";
inline const FilePath pardir = "..";
inline constexpr std::string_view sep = "/";
inline constexpr std::string_view altsep = "";
inline constexpr std::string_view extsep = ".";
inline constexpr std::string_view pathsep = ":";
inline constexpr std::string_view defpath = "/bin:/usr/bin";
inline constexpr std::string_view linesep = "\n";
inline const FilePath devnull = "/dev/null";
#endif

enum class AccessMode : unsigned int
{
    F_OK = 0,
    X_OK = 1,
    W_OK = 2,
    R_OK = 4,
};

[[nodiscard]] constexpr AccessMode operator|(AccessMode lhs, AccessMode rhs) noexcept
{
    return static_cast<AccessMode>(static_cast<unsigned int>(lhs) | static_cast<unsigned int>(rhs));
}

struct StatResult
{
    std::filesystem::file_type type = std::filesystem::file_type::none;
    std::filesystem::perms permissions = std::filesystem::perms::unknown;
    std::uintmax_t size = 0;
    std::uintmax_t hardLinks = 0;
    FileTime atime{};
    FileTime mtime{};
    FileTime ctime{};
};

struct DirEntry
{
    FilePath name;
    FilePath path;
    StatResult info;
    bool isSymlink = false;
};

[[nodiscard]] FilePath getcwd();
void chdir(const FilePath& path);
[[nodiscard]] FilePath executable_path();
[[nodiscard]] FilePath executable_directory();
[[nodiscard]] FilePath temp_directory_path();
[[nodiscard]] std::vector<FilePath> listdir(const FilePath& path = FilePath{"."});
[[nodiscard]] std::vector<DirEntry> scandir(const FilePath& path = FilePath{"."});

void mkdir(const FilePath& path, std::filesystem::perms mode = std::filesystem::perms::all);
void makedirs(const FilePath& path, bool exist_ok = false);
void remove(const FilePath& path);
void unlink(const FilePath& path);
void rmdir(const FilePath& path);
void removedirs(const FilePath& path);
void rename(const FilePath& source, const FilePath& destination);
void renames(const FilePath& source, const FilePath& destination);
void replace(const FilePath& source, const FilePath& destination);
void link(const FilePath& source, const FilePath& destination);
void symlink(const FilePath& source, const FilePath& destination, bool target_is_directory = false);
[[nodiscard]] FilePath readlink(const FilePath& path);

[[nodiscard]] bool access(const FilePath& path, AccessMode mode) noexcept;
[[nodiscard]] StatResult stat(const FilePath& path);
[[nodiscard]] StatResult lstat(const FilePath& path);
void chmod(const FilePath& path, std::filesystem::perms mode);
void truncate(const FilePath& path, std::uintmax_t length);
void utime(const FilePath& path);
void utime(const FilePath& path, FileTime atime, FileTime mtime);

[[nodiscard]] std::optional<std::string> getenv(std::string_view key);
void putenv(std::string_view key, std::string_view value);
void unsetenv(std::string_view key);
// Returns a snapshot. Environment keys are case-insensitive on Windows and case-sensitive on POSIX.
[[nodiscard]] std::unordered_map<std::string, std::string> environ();
[[nodiscard]] std::vector<std::string> get_exec_path();

[[nodiscard]] std::uint32_t getpid() noexcept;
[[nodiscard]] std::uint32_t getppid() noexcept;
[[nodiscard]] std::string getlogin();
[[nodiscard]] unsigned int cpu_count() noexcept;
[[nodiscard]] std::vector<std::byte> urandom(std::size_t size);

namespace path
{

[[nodiscard]] FilePath abspath(const FilePath& value);
[[nodiscard]] FilePath basename(const FilePath& value);
[[nodiscard]] FilePath commonpath(const std::vector<FilePath>& values);
[[nodiscard]] FilePath dirname(const FilePath& value);
[[nodiscard]] bool exists(const FilePath& value) noexcept;
// Expands the current user's "~"; named-user expansion is not supported.
[[nodiscard]] FilePath expanduser(const FilePath& value);
[[nodiscard]] FileTime getatime(const FilePath& value);
[[nodiscard]] FileTime getctime(const FilePath& value);
[[nodiscard]] FileTime getmtime(const FilePath& value);
[[nodiscard]] std::uintmax_t getsize(const FilePath& value);
[[nodiscard]] bool isabs(const FilePath& value) noexcept;
[[nodiscard]] bool isdir(const FilePath& value) noexcept;
[[nodiscard]] bool isfile(const FilePath& value) noexcept;
[[nodiscard]] bool islink(const FilePath& value) noexcept;
[[nodiscard]] bool ismount(const FilePath& value) noexcept;
[[nodiscard]] FilePath join(std::initializer_list<FilePath> values);
[[nodiscard]] bool lexists(const FilePath& value) noexcept;
[[nodiscard]] FilePath normcase(const FilePath& value);
[[nodiscard]] FilePath normpath(const FilePath& value);
[[nodiscard]] FilePath realpath(const FilePath& value);
[[nodiscard]] FilePath relpath(const FilePath& value, const FilePath& start = FilePath{"."});
[[nodiscard]] bool samefile(const FilePath& lhs, const FilePath& rhs);
[[nodiscard]] std::pair<FilePath, FilePath> split(const FilePath& value);
[[nodiscard]] std::pair<FilePath, FilePath> splitdrive(const FilePath& value);
[[nodiscard]] std::pair<FilePath, FilePath> splitext(const FilePath& value);

} // namespace path

} // namespace rad::os

namespace rad
{

[[nodiscard]] std::string PathToUtf8(const os::FilePath& path);

} // namespace rad
