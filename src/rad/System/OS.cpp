#include <rad/System/OS.h>

#include <rad/Core/Unicode.h>

#include <algorithm>
#include <cerrno>
#include <cwchar>
#include <cwctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <thread>

#if defined(RAD_OS_WINDOWS)
#include <windows.h>
#include <bcrypt.h>
#include <io.h>
#include <lmcons.h>
#include <sys/stat.h>
#include <sys/utime.h>
#include <tlhelp32.h>
#ifdef environ
#undef environ
#endif
#else
#include <sys/stat.h>
#include <unistd.h>
#include <utime.h>

#if defined(RAD_OS_MACOS)
#include <mach-o/dyld.h>
#endif

extern char** environ;
#endif

namespace rad::os
{
namespace
{

[[noreturn]] void ThrowError(const std::error_code& error, std::string_view operation)
{
    throw std::system_error(error, std::string(operation));
}

[[noreturn]] void ThrowErrno(std::string_view operation)
{
    ThrowError(std::error_code(errno, std::generic_category()), operation);
}

[[nodiscard]] FileTime FromTimeT(std::time_t value)
{
    return std::chrono::system_clock::from_time_t(value);
}

[[nodiscard]] FileTime ToSystemTime(std::filesystem::file_time_type value)
{
    const auto fileNow = std::filesystem::file_time_type::clock::now();
    const auto systemNow = std::chrono::system_clock::now();
    return std::chrono::time_point_cast<FileTime::duration>(value - fileNow + systemNow);
}

[[nodiscard]] StatResult ReadStat(const FilePath& path, bool followSymlinks)
{
    StatResult result;
    std::error_code error;
    const auto status = followSymlinks ? std::filesystem::status(path, error)
                                       : std::filesystem::symlink_status(path, error);
    if (error)
    {
        ThrowError(error, followSymlinks ? "os::stat" : "os::lstat");
    }
    if (status.type() == std::filesystem::file_type::not_found)
    {
        ThrowError(std::make_error_code(std::errc::no_such_file_or_directory),
                   followSymlinks ? "os::stat" : "os::lstat");
    }

    result.type = status.type();
    result.permissions = status.permissions();

    if (std::filesystem::is_regular_file(status))
    {
        result.size = std::filesystem::file_size(path, error);
        if (error)
        {
            ThrowError(error, followSymlinks ? "os::stat" : "os::lstat");
        }
    }

    error.clear();
    result.hardLinks = std::filesystem::hard_link_count(path, error);
    if (error)
    {
        result.hardLinks = 0;
    }

#if defined(RAD_OS_WINDOWS)
    struct _stat64 nativeStat{};
    if (_wstat64(path.c_str(), &nativeStat) == 0)
    {
        result.atime = FromTimeT(nativeStat.st_atime);
        result.mtime = FromTimeT(nativeStat.st_mtime);
        result.ctime = FromTimeT(nativeStat.st_ctime);
    }
    else
    {
        error.clear();
        const auto modified = std::filesystem::last_write_time(path, error);
        if (!error)
        {
            result.mtime = ToSystemTime(modified);
        }
    }
#else
    struct ::stat nativeStat{};
    const int nativeResult =
        followSymlinks ? ::stat(path.c_str(), &nativeStat) : ::lstat(path.c_str(), &nativeStat);
    if (nativeResult != 0)
    {
        ThrowErrno(followSymlinks ? "os::stat" : "os::lstat");
    }
    result.atime = FromTimeT(nativeStat.st_atime);
    result.mtime = FromTimeT(nativeStat.st_mtime);
    result.ctime = FromTimeT(nativeStat.st_ctime);
#endif
    return result;
}

void ValidateEnvironmentKey(std::string_view key)
{
    if (key.empty() || key.find('=') != std::string_view::npos)
    {
        throw std::invalid_argument("Environment variable names must be non-empty and exclude '='");
    }
}

[[nodiscard]] bool StatusMatches(const FilePath& value, bool useSymlinkStatus,
                                 bool (*predicate)(std::filesystem::file_status)) noexcept
{
    std::error_code error;
    const auto status = useSymlinkStatus ? std::filesystem::symlink_status(value, error)
                                         : std::filesystem::status(value, error);
    return !error && predicate(status);
}

#if !defined(RAD_OS_WINDOWS)
[[nodiscard]] FilePath ReadLinkPath(const char* path)
{
    std::vector<char> buffer(256);
    while (true)
    {
        const auto size = ::readlink(path, buffer.data(), buffer.size());
        if (size < 0)
        {
            ThrowErrno("os::executable_path");
        }
        if (static_cast<std::size_t>(size) < buffer.size())
        {
            return FilePath(std::string(buffer.data(), static_cast<std::size_t>(size)));
        }
        buffer.resize(buffer.size() * 2);
    }
}
#endif

} // namespace

FilePath getcwd()
{
    std::error_code error;
    FilePath result = std::filesystem::current_path(error);
    if (error)
    {
        ThrowError(error, "os::getcwd");
    }
    return result;
}

void chdir(const FilePath& path)
{
    std::error_code error;
    std::filesystem::current_path(path, error);
    if (error)
    {
        ThrowError(error, "os::chdir");
    }
}

FilePath executable_path()
{
#if defined(RAD_OS_WINDOWS)
    std::vector<wchar_t> buffer(256);
    while (true)
    {
        SetLastError(ERROR_SUCCESS);
        const DWORD size =
            GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (size == 0)
        {
            ThrowError(std::error_code(static_cast<int>(GetLastError()), std::system_category()),
                       "os::executable_path");
        }
        if (size < buffer.size())
        {
            return FilePath(std::wstring(buffer.data(), size));
        }
        buffer.resize(buffer.size() * 2);
    }
#elif defined(RAD_OS_LINUX) || defined(RAD_OS_ANDROID)
    return ReadLinkPath("/proc/self/exe");
#elif defined(RAD_OS_MACOS)
    std::uint32_t size = 0;
    if (_NSGetExecutablePath(nullptr, &size) != -1 || size == 0)
    {
        ThrowError(std::make_error_code(std::errc::io_error), "os::executable_path");
    }

    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0)
    {
        ThrowError(std::make_error_code(std::errc::io_error), "os::executable_path");
    }

    std::error_code error;
    FilePath result = std::filesystem::weakly_canonical(FilePath(buffer.data()), error);
    return error ? std::filesystem::absolute(FilePath(buffer.data())) : result;
#elif defined(RAD_OS_FREEBSD)
    return ReadLinkPath("/proc/curproc/file");
#else
    ThrowError(std::make_error_code(std::errc::operation_not_supported), "os::executable_path");
#endif
}

FilePath executable_directory()
{
    return executable_path().parent_path();
}

FilePath temp_directory_path()
{
    std::error_code error;
    FilePath result = std::filesystem::temp_directory_path(error);
    if (error)
    {
        ThrowError(error, "os::temp_directory_path");
    }
    return result;
}

std::vector<FilePath> listdir(const FilePath& path)
{
    std::vector<FilePath> result;
    std::error_code error;
    std::filesystem::directory_iterator iterator(path, error);
    if (error)
    {
        ThrowError(error, "os::listdir");
    }

    for (const auto& entry : iterator)
    {
        result.push_back(entry.path().filename());
    }
    return result;
}

std::vector<DirEntry> scandir(const FilePath& path)
{
    std::vector<DirEntry> result;
    std::error_code error;
    std::filesystem::directory_iterator iterator(path, error);
    if (error)
    {
        ThrowError(error, "os::scandir");
    }

    for (const auto& entry : iterator)
    {
        DirEntry item;
        item.name = entry.path().filename();
        item.path = entry.path();
        item.info = lstat(item.path);
        item.isSymlink = item.info.type == std::filesystem::file_type::symlink;
        result.push_back(std::move(item));
    }
    return result;
}

void mkdir(const FilePath& path, std::filesystem::perms mode)
{
#if defined(RAD_OS_WINDOWS)
    std::error_code error;
    if (!std::filesystem::create_directory(path, error))
    {
        if (!error)
        {
            error = std::make_error_code(std::errc::file_exists);
        }
        ThrowError(error, "os::mkdir");
    }
#else
    if (::mkdir(path.c_str(), static_cast<mode_t>(mode)) != 0)
    {
        ThrowErrno("os::mkdir");
    }
#endif
}

void makedirs(const FilePath& path, bool exist_ok)
{
    std::error_code error;
    if (std::filesystem::create_directories(path, error))
    {
        return;
    }
    if (error)
    {
        ThrowError(error, "os::makedirs");
    }
    if (exist_ok && std::filesystem::is_directory(path, error) && !error)
    {
        return;
    }
    ThrowError(std::make_error_code(std::errc::file_exists), "os::makedirs");
}

void remove(const FilePath& path)
{
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error)
    {
        ThrowError(error, "os::remove");
    }
    if (std::filesystem::is_directory(status))
    {
        ThrowError(std::make_error_code(std::errc::is_a_directory), "os::remove");
    }
    if (!std::filesystem::remove(path, error))
    {
        if (!error)
        {
            error = std::make_error_code(std::errc::no_such_file_or_directory);
        }
        ThrowError(error, "os::remove");
    }
}

void unlink(const FilePath& path)
{
    rad::os::remove(path);
}

void rmdir(const FilePath& path)
{
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error)
    {
        ThrowError(error, "os::rmdir");
    }
    if (!std::filesystem::is_directory(status))
    {
        ThrowError(std::make_error_code(std::errc::not_a_directory), "os::rmdir");
    }
    if (!std::filesystem::remove(path, error))
    {
        if (!error)
        {
            error = std::make_error_code(std::errc::directory_not_empty);
        }
        ThrowError(error, "os::rmdir");
    }
}

void removedirs(const FilePath& path)
{
    rmdir(path);
    FilePath parent = path.parent_path();
    while (!parent.empty())
    {
        std::error_code error;
        if (!std::filesystem::remove(parent, error) || error)
        {
            break;
        }
        parent = parent.parent_path();
    }
}

void rename(const FilePath& source, const FilePath& destination)
{
    std::error_code error;
    std::filesystem::rename(source, destination, error);
    if (error)
    {
        ThrowError(error, "os::rename");
    }
}

void renames(const FilePath& source, const FilePath& destination)
{
    const FilePath parent = destination.parent_path();
    if (!parent.empty())
    {
        makedirs(parent, true);
    }
    rad::os::rename(source, destination);

    FilePath oldParent = source.parent_path();
    while (!oldParent.empty())
    {
        std::error_code error;
        if (!std::filesystem::remove(oldParent, error) || error)
        {
            break;
        }
        oldParent = oldParent.parent_path();
    }
}

void replace(const FilePath& source, const FilePath& destination)
{
#if defined(RAD_OS_WINDOWS)
    if (!MoveFileExW(source.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING))
    {
        ThrowError(std::error_code(static_cast<int>(GetLastError()), std::system_category()),
                   "os::replace");
    }
#else
    rad::os::rename(source, destination);
#endif
}

void link(const FilePath& source, const FilePath& destination)
{
    std::error_code error;
    std::filesystem::create_hard_link(source, destination, error);
    if (error)
    {
        ThrowError(error, "os::link");
    }
}

void symlink(const FilePath& source, const FilePath& destination, bool target_is_directory)
{
    std::error_code error;
    if (target_is_directory)
    {
        std::filesystem::create_directory_symlink(source, destination, error);
    }
    else
    {
        std::filesystem::create_symlink(source, destination, error);
    }
    if (error)
    {
        ThrowError(error, "os::symlink");
    }
}

FilePath readlink(const FilePath& path)
{
    std::error_code error;
    FilePath result = std::filesystem::read_symlink(path, error);
    if (error)
    {
        ThrowError(error, "os::readlink");
    }
    return result;
}

bool access(const FilePath& path, AccessMode mode) noexcept
{
#if defined(RAD_OS_WINDOWS)
    const unsigned int bits = static_cast<unsigned int>(mode);
    const int nativeMode = static_cast<int>(bits & 6U);
    return _waccess(path.c_str(), nativeMode) == 0;
#else
    return ::access(path.c_str(), static_cast<int>(mode)) == 0;
#endif
}

StatResult stat(const FilePath& path)
{
    return ReadStat(path, true);
}

StatResult lstat(const FilePath& path)
{
    return ReadStat(path, false);
}

void chmod(const FilePath& path, std::filesystem::perms mode)
{
    std::error_code error;
    std::filesystem::permissions(path, mode, std::filesystem::perm_options::replace, error);
    if (error)
    {
        ThrowError(error, "os::chmod");
    }
}

void truncate(const FilePath& path, std::uintmax_t length)
{
    std::error_code error;
    std::filesystem::resize_file(path, length, error);
    if (error)
    {
        ThrowError(error, "os::truncate");
    }
}

void utime(const FilePath& path)
{
    const FileTime now = std::chrono::system_clock::now();
    utime(path, now, now);
}

void utime(const FilePath& path, FileTime atime, FileTime mtime)
{
#if defined(RAD_OS_WINDOWS)
    struct __utimbuf64 times{std::chrono::system_clock::to_time_t(atime),
                             std::chrono::system_clock::to_time_t(mtime)};
    if (_wutime64(path.c_str(), &times) != 0)
    {
        ThrowErrno("os::utime");
    }
#else
    struct ::utimbuf times{std::chrono::system_clock::to_time_t(atime),
                           std::chrono::system_clock::to_time_t(mtime)};
    if (::utime(path.c_str(), &times) != 0)
    {
        ThrowErrno("os::utime");
    }
#endif
}

std::optional<std::string> getenv(std::string_view key)
{
    ValidateEnvironmentKey(key);
#if defined(RAD_OS_WINDOWS)
    const std::wstring wideKey = Utf8ToWide(key);
    SetLastError(ERROR_SUCCESS);
    const DWORD required = GetEnvironmentVariableW(wideKey.c_str(), nullptr, 0);
    if (required == 0)
    {
        if (GetLastError() == ERROR_ENVVAR_NOT_FOUND)
        {
            return std::nullopt;
        }
        return std::string{};
    }
    std::wstring value(required, L'\0');
    const DWORD written = GetEnvironmentVariableW(wideKey.c_str(), value.data(), required);
    if (written >= required)
    {
        ThrowError(std::error_code(static_cast<int>(GetLastError()), std::system_category()),
                   "os::getenv");
    }
    value.resize(written);
    return WideToUtf8(value);
#else
    const std::string nativeKey(key);
    const char* value = std::getenv(nativeKey.c_str());
    return value == nullptr ? std::nullopt : std::optional<std::string>{value};
#endif
}

void putenv(std::string_view key, std::string_view value)
{
    ValidateEnvironmentKey(key);
#if defined(RAD_OS_WINDOWS)
    const std::wstring wideKey = Utf8ToWide(key);
    const std::wstring wideValue = Utf8ToWide(value);
    if (!SetEnvironmentVariableW(wideKey.c_str(), wideValue.c_str()))
    {
        ThrowError(std::error_code(static_cast<int>(GetLastError()), std::system_category()),
                   "os::putenv");
    }
#else
    const std::string nativeKey(key);
    const std::string nativeValue(value);
    if (::setenv(nativeKey.c_str(), nativeValue.c_str(), 1) != 0)
    {
        ThrowErrno("os::putenv");
    }
#endif
}

void unsetenv(std::string_view key)
{
    ValidateEnvironmentKey(key);
#if defined(RAD_OS_WINDOWS)
    const std::wstring wideKey = Utf8ToWide(key);
    if (!SetEnvironmentVariableW(wideKey.c_str(), nullptr))
    {
        const DWORD error = GetLastError();
        if (error != ERROR_ENVVAR_NOT_FOUND)
        {
            ThrowError(std::error_code(static_cast<int>(error), std::system_category()),
                       "os::unsetenv");
        }
    }
#else
    const std::string nativeKey(key);
    if (::unsetenv(nativeKey.c_str()) != 0)
    {
        ThrowErrno("os::unsetenv");
    }
#endif
}

std::unordered_map<std::string, std::string> environ()
{
    std::unordered_map<std::string, std::string> result;
#if defined(RAD_OS_WINDOWS)
    const auto freeBlock = [](wchar_t* value)
    {
        if (value != nullptr)
        {
            FreeEnvironmentStringsW(value);
        }
    };
    std::unique_ptr<wchar_t, decltype(freeBlock)> block(GetEnvironmentStringsW(), freeBlock);
    if (block == nullptr)
    {
        ThrowError(std::error_code(static_cast<int>(GetLastError()), std::system_category()),
                   "os::environ");
    }
    for (const wchar_t* item = block.get(); *item != L'\0'; item += std::wcslen(item) + 1)
    {
        std::wstring_view entry(item);
        const std::size_t equals = entry.find(L'=', entry.starts_with(L'=') ? 1U : 0U);
        if (equals != std::wstring_view::npos)
        {
            result.emplace(WideToUtf8(entry.substr(0, equals)),
                           WideToUtf8(entry.substr(equals + 1)));
        }
    }
#else
    for (char** item = ::environ; item != nullptr && *item != nullptr; ++item)
    {
        std::string_view entry(*item);
        const std::size_t equals = entry.find('=');
        if (equals != std::string_view::npos)
        {
            result.emplace(entry.substr(0, equals), entry.substr(equals + 1));
        }
    }
#endif
    return result;
}

std::vector<std::string> get_exec_path()
{
    const auto envPath = getenv("PATH");
    const std::string_view value = envPath ? std::string_view(*envPath) : defpath;

    std::vector<std::string> result;
    std::size_t begin = 0;
    while (begin <= value.size())
    {
        const std::size_t end = value.find(pathsep, begin);
        if (end == std::string_view::npos)
        {
            result.emplace_back(value.substr(begin));
            break;
        }
        result.emplace_back(value.substr(begin, end - begin));
        begin = end + pathsep.size();
    }
    return result;
}

std::uint32_t getpid() noexcept
{
#if defined(RAD_OS_WINDOWS)
    return static_cast<std::uint32_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint32_t>(::getpid());
#endif
}

std::uint32_t getppid() noexcept
{
#if defined(RAD_OS_WINDOWS)
    const DWORD processId = GetCurrentProcessId();
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE)
    {
        return 0;
    }

    PROCESSENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    std::uint32_t parentId = 0;
    if (Process32FirstW(snapshot, &entry))
    {
        do
        {
            if (entry.th32ProcessID == processId)
            {
                parentId = static_cast<std::uint32_t>(entry.th32ParentProcessID);
                break;
            }
        } while (Process32NextW(snapshot, &entry));
    }
    CloseHandle(snapshot);
    return parentId;
#else
    return static_cast<std::uint32_t>(::getppid());
#endif
}

std::string getlogin()
{
#if defined(RAD_OS_WINDOWS)
    std::wstring value(UNLEN + 1, L'\0');
    DWORD size = static_cast<DWORD>(value.size());
    if (!GetUserNameW(value.data(), &size))
    {
        ThrowError(std::error_code(static_cast<int>(GetLastError()), std::system_category()),
                   "os::getlogin");
    }
    value.resize(size > 0 ? size - 1 : 0);
    return WideToUtf8(value);
#else
    long maximum = ::sysconf(_SC_LOGIN_NAME_MAX);
    if (maximum < 0)
    {
        maximum = 256;
    }

    std::vector<char> value(static_cast<std::size_t>(maximum) + 1);
    while (true)
    {
        const int error = ::getlogin_r(value.data(), value.size());
        if (error == 0)
        {
            return value.data();
        }
        if (error != ERANGE)
        {
            ThrowError(std::error_code(error, std::generic_category()), "os::getlogin");
        }
        value.resize(value.size() * 2);
    }
#endif
}

unsigned int cpu_count() noexcept
{
    const unsigned int count = std::thread::hardware_concurrency();
    return count == 0 ? 1U : count;
}

std::vector<std::byte> urandom(std::size_t size)
{
    std::vector<std::byte> result(size);
    if (size == 0)
    {
        return result;
    }
#if defined(RAD_OS_WINDOWS)
    std::size_t offset = 0;
    while (offset < size)
    {
        const auto chunk = static_cast<ULONG>(
            std::min(size - offset, static_cast<std::size_t>(std::numeric_limits<ULONG>::max())));
        const NTSTATUS status =
            BCryptGenRandom(nullptr, reinterpret_cast<PUCHAR>(result.data() + offset), chunk,
                            BCRYPT_USE_SYSTEM_PREFERRED_RNG);
        if (status < 0)
        {
            ThrowError(std::error_code(static_cast<int>(status), std::system_category()),
                       "os::urandom");
        }
        offset += chunk;
    }
#else
    std::ifstream random("/dev/urandom", std::ios::binary);
    if (!random)
    {
        ThrowError(std::make_error_code(std::errc::io_error), "os::urandom");
    }
    random.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(size));
    if (!random)
    {
        ThrowError(std::make_error_code(std::errc::io_error), "os::urandom");
    }
#endif
    return result;
}

namespace path
{

FilePath abspath(const FilePath& value)
{
    std::error_code error;
    FilePath result = std::filesystem::absolute(value, error);
    if (error)
    {
        ThrowError(error, "os::path::abspath");
    }
    return result.lexically_normal();
}

FilePath basename(const FilePath& value)
{
    return value.filename();
}

FilePath commonpath(const std::vector<FilePath>& values)
{
    if (values.empty())
    {
        throw std::invalid_argument("os::path::commonpath requires at least one path");
    }

    const bool absolute = values.front().is_absolute();
    const FilePath drive = normcase(splitdrive(values.front()).first);
    std::vector<std::vector<FilePath>> components;
    components.reserve(values.size());
    for (const FilePath& value : values)
    {
        if (value.is_absolute() != absolute)
        {
            throw std::invalid_argument(
                "os::path::commonpath cannot mix absolute and relative paths");
        }
        if (normcase(splitdrive(value).first) != drive)
        {
            throw std::invalid_argument("os::path::commonpath cannot mix different drives");
        }
        const FilePath normalized = normpath(value);
        components.emplace_back(normalized.begin(), normalized.end());
    }

    FilePath result;
    for (std::size_t component = 0; component < components.front().size(); ++component)
    {
        for (std::size_t index = 1; index < components.size(); ++index)
        {
            if (component >= components[index].size() ||
                normcase(components[index][component]) != normcase(components.front()[component]))
            {
                return result;
            }
        }
        result /= components.front()[component];
    }
    return result;
}

FilePath dirname(const FilePath& value)
{
    return value.parent_path();
}

bool exists(const FilePath& value) noexcept
{
    return StatusMatches(value, false, std::filesystem::exists);
}

FilePath expanduser(const FilePath& value)
{
    const auto& native = value.native();
    using Char = FilePath::value_type;
    if (native.empty() || native.front() != static_cast<Char>('~'))
    {
        return value;
    }
    if (native.size() > 1 && native[1] != static_cast<Char>('/') &&
        native[1] != static_cast<Char>('\\'))
    {
        return value;
    }

    FilePath home;
#if defined(RAD_OS_WINDOWS)
    if (const auto profile = getenv("USERPROFILE"))
    {
        home = FilePath(Utf8ToWide(*profile));
    }
    else
    {
        const auto drive = getenv("HOMEDRIVE");
        const auto directory = getenv("HOMEPATH");
        if (!drive || !directory)
        {
            return value;
        }
        home = FilePath(Utf8ToWide(*drive + *directory));
    }
#else
    const auto directory = getenv("HOME");
    if (!directory)
    {
        return value;
    }
    home = FilePath(*directory);
#endif

    if (native.size() == 1)
    {
        return home;
    }
    return home / FilePath(native.substr(1)).relative_path();
}

FileTime getatime(const FilePath& value)
{
    return stat(value).atime;
}

FileTime getctime(const FilePath& value)
{
    return stat(value).ctime;
}

FileTime getmtime(const FilePath& value)
{
    return stat(value).mtime;
}

std::uintmax_t getsize(const FilePath& value)
{
    std::error_code error;
    const std::uintmax_t result = std::filesystem::file_size(value, error);
    if (error)
    {
        ThrowError(error, "os::path::getsize");
    }
    return result;
}

bool isabs(const FilePath& value) noexcept
{
    return value.is_absolute();
}

bool isdir(const FilePath& value) noexcept
{
    return StatusMatches(value, false, std::filesystem::is_directory);
}

bool isfile(const FilePath& value) noexcept
{
    return StatusMatches(value, false, std::filesystem::is_regular_file);
}

bool islink(const FilePath& value) noexcept
{
    return StatusMatches(value, true, std::filesystem::is_symlink);
}

bool ismount(const FilePath& value) noexcept
{
#if defined(RAD_OS_WINDOWS)
    std::error_code error;
    const FilePath absolute = std::filesystem::absolute(value, error).lexically_normal();
    if (error)
    {
        return false;
    }

    std::wstring volumePath(32768, L'\0');
    if (!GetVolumePathNameW(absolute.c_str(), volumePath.data(),
                            static_cast<DWORD>(volumePath.size())))
    {
        return false;
    }
    volumePath.resize(std::wcslen(volumePath.c_str()));
    return normcase(normpath(absolute)) == normcase(normpath(FilePath(volumePath)));
#else
    struct ::stat pathInfo{};
    struct ::stat parentInfo{};
    if (::lstat(value.c_str(), &pathInfo) != 0 || S_ISLNK(pathInfo.st_mode))
    {
        return false;
    }
    const FilePath parent = value / "..";
    if (::lstat(parent.c_str(), &parentInfo) != 0)
    {
        return false;
    }
    return pathInfo.st_dev != parentInfo.st_dev ||
           (pathInfo.st_dev == parentInfo.st_dev && pathInfo.st_ino == parentInfo.st_ino);
#endif
}

FilePath join(std::initializer_list<FilePath> values)
{
    FilePath result;
    for (const FilePath& value : values)
    {
        result /= value;
    }
    return result;
}

bool lexists(const FilePath& value) noexcept
{
    std::error_code error;
    const auto status = std::filesystem::symlink_status(value, error);
    return !error && status.type() != std::filesystem::file_type::not_found;
}

FilePath normcase(const FilePath& value)
{
#if defined(RAD_OS_WINDOWS)
    std::wstring native = value.native();
    std::transform(native.begin(), native.end(), native.begin(), [](wchar_t character)
                   { return character == L'/' ? L'\\' : std::towlower(character); });
    return FilePath(std::move(native));
#else
    return value;
#endif
}

FilePath normpath(const FilePath& value)
{
    if (value.empty())
    {
        return FilePath{"."};
    }
    FilePath result = value.lexically_normal();
    if (result.filename().empty() && result != result.root_path())
    {
        result = result.parent_path();
    }
    return result;
}

FilePath realpath(const FilePath& value)
{
    std::error_code error;
    const FilePath absolute = std::filesystem::absolute(value, error);
    if (error)
    {
        ThrowError(error, "os::path::realpath");
    }
    FilePath result = std::filesystem::weakly_canonical(absolute, error);
    if (error)
    {
        ThrowError(error, "os::path::realpath");
    }
    return result;
}

FilePath relpath(const FilePath& value, const FilePath& start)
{
    std::error_code error;
    FilePath result = std::filesystem::relative(value, start, error);
    if (error)
    {
        ThrowError(error, "os::path::relpath");
    }
    return result;
}

bool samefile(const FilePath& lhs, const FilePath& rhs)
{
    std::error_code error;
    const bool result = std::filesystem::equivalent(lhs, rhs, error);
    if (error)
    {
        ThrowError(error, "os::path::samefile");
    }
    return result;
}

std::pair<FilePath, FilePath> split(const FilePath& value)
{
    return {value.parent_path(), value.filename()};
}

std::pair<FilePath, FilePath> splitdrive(const FilePath& value)
{
#if defined(RAD_OS_WINDOWS)
    const std::wstring& native = value.native();
    const auto isSeparator = [](wchar_t character)
    { return character == L'\\' || character == L'/'; };
    if (native.size() >= 2 && isSeparator(native[0]) && isSeparator(native[1]))
    {
        std::size_t serverStart = 2;
        if (native.size() >= 8 && native[2] == L'?' && isSeparator(native[3]) &&
            std::towupper(native[4]) == L'U' && std::towupper(native[5]) == L'N' &&
            std::towupper(native[6]) == L'C' && isSeparator(native[7]))
        {
            serverStart = 8;
        }

        const std::size_t serverEnd = native.find_first_of(L"\\/", serverStart);
        if (serverEnd != std::wstring::npos && serverEnd + 1 < native.size())
        {
            const std::size_t shareEnd = native.find_first_of(L"\\/", serverEnd + 1);
            const std::size_t driveEnd = shareEnd == std::wstring::npos ? native.size() : shareEnd;
            return {FilePath(native.substr(0, driveEnd)), FilePath(native.substr(driveEnd))};
        }
    }
#endif
    const FilePath drive = value.root_name();
    const auto& nativeValue = value.native();
    return {drive, FilePath(nativeValue.substr(drive.native().size()))};
}

std::pair<FilePath, FilePath> splitext(const FilePath& value)
{
    const auto& filename = value.filename().native();
    using Char = FilePath::value_type;
    const std::size_t dot = filename.rfind(static_cast<Char>('.'));
    if (dot == FilePath::string_type::npos)
    {
        return {value, FilePath{}};
    }

    const bool hasExtension =
        std::any_of(filename.begin(), filename.begin() + static_cast<std::ptrdiff_t>(dot),
                    [](Char character) { return character != static_cast<Char>('.'); });
    if (!hasExtension)
    {
        return {value, FilePath{}};
    }

    FilePath root = value;
    root.replace_filename(FilePath(filename.substr(0, dot)));
    return {std::move(root), FilePath(filename.substr(dot))};
}

} // namespace path

} // namespace rad::os

namespace rad
{

std::string PathToUtf8(const os::FilePath& path)
{
#if defined(RAD_OS_WINDOWS)
    return WideToUtf8(path.native());
#else
    return path.string();
#endif
}

} // namespace rad
