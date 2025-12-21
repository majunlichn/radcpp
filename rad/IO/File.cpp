#include <rad/IO/File.h>
#include <cassert>
#include <cerrno>
#include <cstdarg>
#include <ctime>

#ifdef _WIN32
#include <Windows.h>
#ifdef CreateDirectory
#undef CreateDirectory
#endif
#ifdef CreateHardLink
#undef CreateHardLink
#endif
#ifdef CopyFile
#undef CopyFile
#endif
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace rad
{

FilePath MakeFilePath(std::string_view str)
{
    return FilePath((const char8_t*)str.data());
}

std::string ToString(const FilePath& path)
{
    return std::string((const char*)path.u8string().c_str());
}

void Swap(FilePath& lhs, FilePath& rhs)
{
    std::filesystem::swap(lhs, rhs);
}

size_t Hash(const FilePath& p)
{
    return std::filesystem::hash_value(p);
}

FilePath GetCurrentPath()
{
    return std::filesystem::current_path();
}

FilePath GetWorkingDirectory()
{
    return std::filesystem::current_path();
}

void SetCurrentPath(const FilePath& p)
{
    std::filesystem::current_path(p);
}

FilePath GetTempDirectory()
{
    return std::filesystem::temp_directory_path();
}

FilePath MakeAbsolute(const FilePath& p)
{
    return std::filesystem::absolute(p);
}

FilePath MakeRelative(const FilePath& p, const FilePath& base)
{
    return std::filesystem::relative(p, base);
}

FilePath MakeProximate(const FilePath& p, const FilePath& base)
{
    return std::filesystem::proximate(p, base);
}

FilePath MakeCanonical(const FilePath& p)
{
    return std::filesystem::canonical(p);
}

FilePath MakeWeaklyCanonical(const FilePath& p)
{
    return std::filesystem::weakly_canonical(p);
}

bool Exists(const FilePath& p)
{
    return std::filesystem::exists(p);
}

bool IsEquivalent(const FilePath& p1, const FilePath& p2)
{
    return std::filesystem::equivalent(p1, p2);
}

std::uint64_t GetFileSize(const FilePath& p)
{
    return std::filesystem::file_size(p);
}

FileStatus GetFileStatus(const FilePath& p)
{
    return std::filesystem::status(p);
}

FileStatus GetSymlinkStatus(const FilePath& p)
{
    return std::filesystem::symlink_status(p);
}

bool Exists(FileStatus s)
{
    return std::filesystem::exists(s);
}

std::uint64_t GetHardLinkCount(const FilePath& p)
{
    return std::filesystem::hard_link_count(p);
}

FileTime GetLastWriteTime(const FilePath& p)
{
    return std::filesystem::last_write_time(p);
}

void SetLastWriteTime(const FilePath& p, FileTime t)
{
    std::filesystem::last_write_time(p, t);
}

void SetPermissions(const FilePath& p, FilePerms permissions, FilePermOptions options)
{
    std::filesystem::permissions(p, permissions, options);
}

FilePath ReadSymlink(const FilePath& p)
{
    return std::filesystem::read_symlink(p);
}

bool CreateDirectory(const FilePath& p)
{
    return std::filesystem::create_directory(p);
}

bool CreateDirectory(const FilePath& p, const FilePath& copyAttribs)
{
    return std::filesystem::create_directory(p, copyAttribs);
}

bool CreateDirectories(const FilePath& p)
{
    return std::filesystem::create_directories(p);
}

void CreateHardLink(const FilePath& target, const FilePath& link)
{
    return std::filesystem::create_hard_link(target, link);
}

void CreateSymlink(const FilePath& target, const FilePath& link)
{
    return std::filesystem::create_symlink(target, link);
}

void CreateDirectorySymlink(const FilePath& target, const FilePath& link)
{
    return std::filesystem::create_directory_symlink(target, link);
}

void Copy(const FilePath& from, const FilePath& to, FileCopyOptions options)
{
    std::filesystem::copy(from, to, options);
}

void CopyFile(const FilePath& from, const FilePath& to, FileCopyOptions options)
{
    std::filesystem::copy_file(from, to, options);
}

void CopySymlink(const FilePath& from, const FilePath& to)
{
    std::filesystem::copy_symlink(from, to);
}

bool Remove(const FilePath& p)
{
    return std::filesystem::remove(p);
}

std::uint64_t RemoveAll(const FilePath& p)
{
    return std::filesystem::remove_all(p);
}

void Rename(const FilePath& oldPath, const FilePath& newPath)
{
    std::filesystem::rename(oldPath, newPath);
}

void ResizeFile(const FilePath& p, std::uint64_t newSize)
{
    std::filesystem::resize_file(p, newSize);
}

// Return the space info, the same as POSIX statvfs.
FileSpaceInfo GetSpaceInfo(const FilePath& p)
{
    return std::filesystem::space(p);
}

bool IsBlockFile(const FilePath& p)
{
    return std::filesystem::is_block_file(p);
}

bool IsCharacterFile(const FilePath& p)
{
    return std::filesystem::is_character_file(p);
}

bool IsDirectory(const FilePath& p)
{
    return std::filesystem::is_directory(p);
}

bool IsEmpty(const FilePath& p)
{
    return std::filesystem::is_empty(p);
}

bool IsFIFO(const FilePath& p)
{
    return std::filesystem::is_fifo(p);
}

bool IsOther(const FilePath& p)
{
    return std::filesystem::is_other(p);
}

bool IsRegularFile(const FilePath& p)
{
    return std::filesystem::is_regular_file(p);
}

bool IsSocket(const FilePath& p)
{
    return std::filesystem::is_socket(p);
}

bool IsSymlink(const FilePath& p)
{
    return std::filesystem::is_symlink(p);
}

bool IsStatusKnown(const FileStatus& s)
{
    return std::filesystem::status_known(s);
}

std::string GetFilePathModifiedTimeString(const FilePath& path, std::string_view format)
{
    std::tm dateTime = {};

#if defined(RAD_OS_WINDOWS)
    struct __stat64 fileStatus = {};
    _wstat64(path.c_str(), &fileStatus);
    localtime_s(&dateTime, &fileStatus.st_mtime);
#else
    struct stat fileStatus = {};
    stat(path.c_str(), &fileStatus);
    localtime_r(&fileStatus.st_mtime, &dateTime);
#endif
    std::string buffer(128, 0);
    size_t bytesWritten = strftime(buffer.data(), buffer.size(), format.data(), &dateTime);
    if (bytesWritten == 0)
    {
        buffer.resize(1024);
        bytesWritten = strftime(buffer.data(), buffer.size(), format.data(), &dateTime);
    }
    if (bytesWritten != 0)
    {
        buffer.resize(bytesWritten);
    }
    else
    {
        buffer.clear();
    }
    return buffer;
}


File::File()
{
}

File::~File()
{
    if (m_handle)
    {
        Close();
    }
}

bool File::Open(std::string_view fileName, std::string_view mode)
{
#if defined(RAD_OS_WINDOWS)
    errno_t err = fopen_s(&m_handle, fileName.data(), mode.data());
#else
    m_handle = fopen(fileName.data(), mode.data());
    error_t err = 0;
    if (m_handle == nullptr)
    {
        err = errno;
    }
#endif
    if ((err == 0) && (m_handle != nullptr))
    {
        m_path = fileName;
        return true;
    }
    else
    {
        m_handle = nullptr;
        return false;
    }
}

void File::Close()
{
    fclose(m_handle);
    m_handle = nullptr;
}

bool File::IsOpen()
{
    return (m_handle != nullptr);
}

void File::Flush()
{
    fflush(m_handle);
}

size_t File::Read(void* buffer, size_t elementSize, size_t elementCount)
{
    return fread(buffer, elementSize, elementCount, m_handle);
}

size_t File::ReadLine(void* buffer, size_t bufferSize)
{
    size_t bytesRead = 0;
    char* str = static_cast<char*>(buffer);
    while (bytesRead < bufferSize)
    {
        int32_t c = GetChar();
        if ((c == '\n') || (c == EOF))
        {
            break;
        }
        str[bytesRead] = static_cast<char>(c);
        bytesRead++;
    }

    const size_t endIndex = ((bytesRead < bufferSize) ? bytesRead : (bufferSize - 1));
    str[endIndex] = '\0';

    return bytesRead;
}

size_t File::ReadLine(std::string& buffer)
{
    size_t bytesRead = 0;
    while (true)
    {
        int32_t c = GetChar();
        if ((c == '\n') || (c == EOF))
        {
            break;
        }
        buffer.push_back(static_cast<char>(c));
        bytesRead++;
    }
    return bytesRead;
}

size_t File::Write(const void* buffer, size_t elementSize, size_t elementCount)
{
    return fwrite(buffer, elementSize, elementCount, m_handle);
}

int File::Print(const char* format, ...)
{
    int ret = 0;
    va_list args;
    va_start(args, format);
    ret = vfprintf(m_handle, format, args);
    va_end(args);
    return ret;
}

int32_t File::GetChar()
{
    return std::getc(m_handle);
}

int64_t File::Seek(int64_t offset, int origin)
{
    return fseek(m_handle, static_cast<long>(offset), origin);
}

int64_t File::Rseek(int64_t offset)
{
    return fseek(m_handle, static_cast<long>(offset), SEEK_END);
}

void File::Rewind()
{
    std::rewind(m_handle);
}

void File::FastForward()
{
    Rseek(0);
}

bool File::GetInfo(std::string_view fileName, FileInfo* pInfo)
{
    assert(pInfo != nullptr);
#if defined(RAD_OS_WINDOWS)
    struct _stat64 status {};
    const int ret = _stat64(fileName.data(), &status);
#else
    struct stat64 status {};
    const int ret = stat64(fileName.data(), &status);
#endif
    if (ret != 0)
    {
        return false;
    }

    static_assert(sizeof(FileInfo::size) == sizeof(status.st_size));

    if (pInfo)
    {
        pInfo->size = static_cast<decltype(FileInfo::size)>(status.st_size);
        pInfo->ctime = static_cast<decltype(FileInfo::ctime)>(status.st_ctime);
        pInfo->atime = static_cast<decltype(FileInfo::atime)>(status.st_atime);
        pInfo->mtime = static_cast<decltype(FileInfo::mtime)>(status.st_mtime);
    }

    return true;
}

bool File::GetInfo(FileInfo* pInfo)
{
    return GetInfo(m_path, pInfo);
}

uint64_t File::GetSize()
{
    FileInfo status = {};
    if (GetInfo(&status))
    {
        return status.size;
    }
    return 0;
}

int64_t File::Tell()
{
    return ftell(m_handle);
}

bool File::IsEndReached()
{
    return (feof(m_handle) != 0);
}

std::string File::ReadAll(std::string_view path)
{
    File file;
    std::string buffer;
    if (file.Open(path, "rb"))
    {
        int64_t fileSize = file.GetSize();
        buffer.resize(fileSize);
        file.Read(buffer.data(), fileSize);
        return buffer;
    }
    return buffer;
}

std::vector<std::string> File::ReadLines(std::string_view path)
{
    File file;
    std::vector<std::string> lines;
    if (file.Open(path, "r"))
    {
        std::string line;
        while (true)
        {
            int32_t c = file.GetChar();
            if ((c != '\n') && (c != EOF))
            {
                line.push_back(static_cast<char>(c));
            }
            else
            {
                lines.push_back(line);
                line.clear();
                if (c == EOF)
                {
                    break;
                }
            }
        }
    }
    return lines;
}

} // namespace rad
