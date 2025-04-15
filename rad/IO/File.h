#pragma once

#include <rad/Core/Platform.h>
#include <filesystem>

namespace rad
{

// Encoding: char on POSIX, wchar_t on Windows.
using FilePath = std::filesystem::path;

// Treat std::string as UTF-8 encoded.
FilePath MakeFilePath(std::string_view str);
std::string ToString(const FilePath& path);

void Swap(FilePath& lhs, FilePath& rhs);
size_t Hash(const FilePath& p);

using FileSystemError = std::filesystem::filesystem_error;

using DirectoryEntry = std::filesystem::directory_entry;
using DirectoryIterator = std::filesystem::directory_iterator;
using RecursiveDirectorIterator = std::filesystem::recursive_directory_iterator;

using FileStatus = std::filesystem::file_status;
using FileSpaceInfo = std::filesystem::space_info;

using FileType = std::filesystem::file_type;
using FilePerms = std::filesystem::perms;
using FilePermOptions = std::filesystem::perm_options;
using FileCopyOptions = std::filesystem::copy_options;
using DirectoryOptions = std::filesystem::directory_options;

using FileTime = std::filesystem::file_time_type;

FilePath GetCurrentPath();
FilePath GetWorkingDirectory();
void SetCurrentPath(const FilePath& p);
FilePath GetTempDirectory();

FilePath MakeAbsolute(const FilePath& p);
FilePath MakeRelative(const FilePath& p, const FilePath& base);
FilePath MakeProximate(const FilePath& p, const FilePath& base);
FilePath MakeCanonical(const FilePath& p);
FilePath MakeWeaklyCanonical(const FilePath& p);

bool Exists(const FilePath& p);
bool IsEquivalent(const FilePath& p1, const FilePath& p2);

std::uint64_t GetFileSize(const FilePath& p);
FileStatus GetFileStatus(const FilePath& p);
FileStatus GetSymlinkStatus(const FilePath& p);
bool Exists(FileStatus s);
std::uint64_t GetHardLinkCount(const FilePath& p);
FileTime GetLastWriteTime(const FilePath& p);
void SetLastWriteTime(const FilePath& p, FileTime t);

void SetPermissions(const FilePath& p, FilePerms permissions,
    FilePermOptions options = FilePermOptions::replace);

FilePath ReadSymlink(const FilePath& p);

bool CreateDirectory(const FilePath& p);
bool CreateDirectory(const FilePath& p, const FilePath& copyAttribs);
bool CreateDirectories(const FilePath& p);

void CreateHardLink(const FilePath& target, const FilePath& link);
void CreateSymlink(const FilePath& target, const FilePath& link);
void CreateDirectorySymlink(const FilePath& target, const FilePath& link);

void Copy(const FilePath& from, const FilePath& to, FileCopyOptions options = FileCopyOptions::none);
void CopyFile(const FilePath& from, const FilePath& to, FileCopyOptions options = FileCopyOptions::none);
void CopySymlink(const FilePath& from, const FilePath& to);

// Delete file or empty directory, the same as POSIX remove.
// Symlinks are not followed (symlink is removed, not its target).
// Return true if the file was deleted, false if it did not exist.
bool Remove(const FilePath& p);

// Delete file or directory with all its subdirectories, recursively.
// Symlinks are not followed (symlink is removed, not its target).
// Return the number of files and directories that were deleted.
std::uint64_t RemoveAll(const FilePath& p);

void Rename(const FilePath& oldPath, const FilePath& newPath);

void ResizeFile(const FilePath& p, std::uint64_t newSize);

// The same as POSIX statvfs:
// info.capacity is set as if by f_blocks * f_frsize.
// info.free is set to f_bfree * f_frsize.
// info.available is set to f_bavail * f_frsize.
// Any member that could not be determined is set to static_cast<std::uint64_t>(-1).
FileSpaceInfo GetSpaceInfo(const FilePath& p);

// File types
bool IsBlockFile(const FilePath& p);
bool IsCharacterFile(const FilePath& p);
bool IsDirectory(const FilePath& p);
bool IsEmpty(const FilePath& p);
bool IsFIFO(const FilePath& p);
bool IsOther(const FilePath& p);
bool IsRegularFile(const FilePath& p);
bool IsSocket(const FilePath& p);
bool IsSymlink(const FilePath& p);
bool IsStatusKnown(const FileStatus& s);

std::string GetFilePathModifiedTimeString(const FilePath& path, std::string_view format = "%F %T");

struct FileInfo
{
    uint64_t  size;     // Size of the file in bytes.
    uint64_t  ctime;    // Time of creation of the file (not valid on FAT).
    uint64_t  atime;    // Time of last access to the file (not valid on FAT).
    uint64_t  mtime;    // Time of last modification to the file.
};

class File
{
public:
    File();
    ~File();

    const std::string& GetPath() const { return m_path; }
    std::FILE* GetHandle() const { return m_handle; }

    // Read = "r"/"rb"
    // Write = "w"/"wb"
    // Append = "a"/"ab"
    // ReadWrite = "w+"/"wb+R"
    // ReadWriteExist = "r+"/"rb+R"
    // ReadAppend (no overwrite to existing data) = "a+"/"ab+R"
    bool Open(std::string_view fileName, std::string_view mode);
    void Close();
    bool IsOpen();
    void Flush();

    size_t Read(void* buffer, size_t elementSize, size_t elementCount = 1);
    size_t ReadLine(void* buffer, size_t bufferSize);
    size_t ReadLine(std::string& buffer);
    size_t Write(const void* buffer, size_t elementSize, size_t elementCount = 1);

    int Print(const char* format, ...);

    int32_t GetChar();

    // Sets the file position indicator for the file stream stream to the value pointed to by offset.
    // @origin: any of RW_SEEK_SET, RW_SEEK_CUR, RW_SEEK_END;
    int64_t Seek(int64_t offset, int origin);
    int64_t Rseek(int64_t offset);
    void Rewind();
    void FastForward();

    static bool GetInfo(std::string_view fileName, FileInfo* pInfo);
    bool GetInfo(FileInfo* pInfo);
    uint64_t GetSize();
    int64_t Tell();
    bool IsEndReached();

    static std::string ReadAll(std::string_view path);
    static std::vector<std::string> ReadLines(std::string_view path);

private:
    std::string m_path;
    std::FILE* m_handle = nullptr;

}; // class File

} // namespace rad
