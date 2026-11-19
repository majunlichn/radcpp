#pragma once

#include <rad/Core/Result.h>
#include <rad/Core/Span.h>
#include <rad/System/FileSystem.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace rad
{

enum class FileAccess
{
    Read,
    Write,
    ReadWrite,
    Append, // Write only; writes go to the end regardless of the current position.
};

enum class FileOpenDisposition
{
    OpenExisting,
    OpenOrCreate,
    CreateNew,        // Fail if the path already exists.
    CreateAlways,     // Create or truncate in place.
    TruncateExisting, // Must exist; truncate in place.
};

enum class FileSeekOrigin
{
    Begin,
    Current,
    End,
};

using Sha256Digest = std::array<std::byte, 32>;

// A move-only RAII wrapper for byte I/O. Concurrent access to one File is not supported.
// Compare error codes directly with std::errc conditions (e.g. ec == std::errc::file_exists).
// Other I/O failures may carry platform-specific codes with no matching std::errc condition.
// Read, ReadLine, Write, Seek, Position, Size(), and Flush report bad_file_descriptor when
// the file is closed. Read, ReadLine, and Write also report it if access disallows the operation.
class File
{
public:
    File() noexcept = default;
    ~File() noexcept;

    File(const File&) = delete;
    File& operator=(const File&) = delete;

    File(File&& other) noexcept;
    File& operator=(File&& other) noexcept;

    // Always binary. Read requires OpenExisting; Append does not allow truncation.
    // An empty path, embedded NUL, or invalid access/disposition returns invalid_argument
    // without closing the current file. Once reopening starts, any failure leaves it closed.
    // For ordinary file paths, OpenExisting and TruncateExisting report no_such_file_or_directory
    // for a missing target, and CreateNew reports file_exists for a name collision. Other
    // failures, such as permission or sharing violations, can take precedence.
    // Directories cannot be opened; the error condition may differ by platform.
    [[nodiscard]] Result<void> Open(
        const FilePath& path, FileAccess access,
        FileOpenDisposition disposition = FileOpenDisposition::OpenExisting) noexcept;
    // Closes the file even on failure. Some buffered write errors are first detected here.
    // A successful close does not undo errors reported by earlier calls.
    [[nodiscard]] Result<void> Close() noexcept;

    [[nodiscard]] bool IsOpen() const noexcept;
    [[nodiscard]] const FilePath& Path() const noexcept;
    [[nodiscard]] FileAccess Access() const noexcept;

    // Returns the number of bytes read. A short read at EOF is successful.
    // Empty Read and Write requests validate access but do not flush or seek.
    // On error the buffer and position may already have changed.
    [[nodiscard]] Result<std::size_t> Read(Span<std::byte> destination) noexcept;
    // For the pointer overloads of Read and Write, nullptr with nonzero size is invalid_argument.
    [[nodiscard]] Result<std::size_t> Read(void* destination, std::size_t size) noexcept;
    // A failed write may have written part of the source; do not blindly retry it.
    [[nodiscard]] Result<void> Write(Span<const std::byte> source) noexcept;
    [[nodiscard]] Result<void> Write(const void* source, std::size_t size) noexcept;
    // Returns true for a line, including an empty line. Splits on LF and strips one trailing CR.
    // On false, line is cleared and ec distinguishes normal EOF (clear) from an error.
    // ec is also cleared on success.
    [[nodiscard]] bool ReadLine(std::string& line, std::error_code& ec) noexcept;
    // Throws std::system_error on failure; false means normal EOF.
    [[nodiscard]] bool ReadLine(std::string& line);

    // Flushes pending writes before seeking. An unrecognized origin returns invalid_argument.
    [[nodiscard]] Result<void> Seek(std::int64_t offset,
                                   FileSeekOrigin origin = FileSeekOrigin::Begin) noexcept;
    [[nodiscard]] Result<std::uint64_t> Position() noexcept;
    // Size in bytes of the opened file, even if its path has since been replaced.
    // Flushes pending writes first and may return a write error.
    // Reports operation_not_supported for a non-regular file and value_too_large if the
    // size cannot be represented.
    [[nodiscard]] Result<std::uint64_t> Size() noexcept;
    // Flushes buffered writes to the OS without guaranteeing durability. A later successful
    // flush does not recover data lost by an earlier failure.
    [[nodiscard]] Result<void> Flush() noexcept;

    // True after a read encounters EOF; does not predict whether the next read will.
    [[nodiscard]] bool Eof() const noexcept;

    [[nodiscard]] static Result<std::vector<std::byte>> ReadAllBytes(
        const FilePath& path) noexcept;
    // Returns the raw bytes as a string, without encoding validation or newline conversion.
    [[nodiscard]] static Result<std::string> ReadAllText(const FilePath& path) noexcept;
    [[nodiscard]] static Result<std::vector<std::string>> ReadAllLines(
        const FilePath& path) noexcept;

    // Create or atomically replace a regular file after writing all content.
    // Symlinks and other non-regular destinations are rejected. Replacement does not
    // preserve file identity or metadata, and does not guarantee crash durability.
    // The new file uses inherited default permissions on Windows and at most 0600 on POSIX.
    // Use Open(path, FileAccess::Write, FileOpenDisposition::CreateAlways) to truncate in place.
    // Use this in directories you control. On Windows, an open target may prevent replacement;
    // on POSIX, an open handle can continue to refer to the original file after replacement.
    // If the destination check finds a directory, returns is_a_directory; other non-regular
    // destinations and invalid path arguments return invalid_argument.
    // A file_exists error can refer to a temporary name, not to the destination.
    [[nodiscard]] static Result<void> WriteBytes(const FilePath& path,
                                                 Span<const std::byte> content) noexcept;
    [[nodiscard]] static Result<void> WriteText(const FilePath& path,
                                                std::string_view content) noexcept;
    // Appends without guaranteeing that one call is atomic against concurrent writers.
    [[nodiscard]] static Result<void> AppendBytes(const FilePath& path,
                                                  Span<const std::byte> content) noexcept;
    [[nodiscard]] static Result<void> AppendText(const FilePath& path,
                                                 std::string_view content) noexcept;

    [[nodiscard]] static bool Exists(const FilePath& path) noexcept;
    [[nodiscard]] static bool IsRegularFile(const FilePath& path) noexcept;
    // Size in bytes of the file currently at path. An empty path or embedded NUL returns
    // invalid_argument; an unrepresentable size returns value_too_large.
    [[nodiscard]] static Result<std::uint64_t> Size(const FilePath& path) noexcept;
    [[nodiscard]] static std::optional<FileTime> LastWriteTime(const FilePath& path) noexcept;
    // Fast checksums for comparison and integrity checks; neither is cryptographically secure.
    [[nodiscard]] static std::optional<std::uint32_t> Crc32(const FilePath& path) noexcept;
    [[nodiscard]] static std::optional<std::uint64_t> XxHash64(const FilePath& path) noexcept;
    [[nodiscard]] static std::optional<Sha256Digest> Sha256(const FilePath& path) noexcept;
    [[nodiscard]] static bool Remove(const FilePath& path) noexcept;
    [[nodiscard]] static bool Copy(const FilePath& source, const FilePath& destination,
                                   bool overwrite = false) noexcept;
    [[nodiscard]] static bool Move(const FilePath& source, const FilePath& destination) noexcept;

private:
    enum class StreamDirection
    {
        None,
        Read,
        Write,
    };

    // Requires a closed File. Takes ownership of the descriptor even on failure.
    [[nodiscard]] Result<void> AdoptDescriptor(int descriptor, FilePath path,
                                               FileAccess access) noexcept;
    [[nodiscard]] Result<void> PrepareRead(bool transferBytes = true) noexcept;
    [[nodiscard]] Result<void> PrepareWrite(bool transferBytes) noexcept;

    std::FILE* m_stream = nullptr;
    FilePath m_path;
    FileAccess m_access = FileAccess::Read;
    StreamDirection m_streamDirection = StreamDirection::None;
}; // class File

} // namespace rad
