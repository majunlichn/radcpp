#pragma once

#include <rad/Core/Span.h>
#include <rad/System/FileSystem.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rad
{

enum class FileMode
{
    Read,
    Write,
    ReadWrite,
    Append,
};

enum class FileSeekOrigin
{
    Begin,
    Current,
    End,
};

using Sha256Digest = std::array<std::byte, 32>;

// A small, move-only RAII wrapper for common file operations.
class File
{
public:
    File() noexcept = default;
    ~File() noexcept;

    File(const File&) = delete;
    File& operator=(const File&) = delete;

    File(File&& other) noexcept;
    File& operator=(File&& other) noexcept;

    [[nodiscard]] bool Open(const FilePath& path, FileMode mode) noexcept;
    [[nodiscard]] bool Close() noexcept;

    [[nodiscard]] bool IsOpen() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const FilePath& Path() const noexcept;
    [[nodiscard]] FileMode Mode() const noexcept;

    // Returns the number of bytes read. A short read at EOF is successful.
    [[nodiscard]] std::size_t Read(Span<std::byte> destination) noexcept;
    [[nodiscard]] std::size_t Read(void* destination, std::size_t size) noexcept;
    [[nodiscard]] bool Write(Span<const std::byte> source) noexcept;
    [[nodiscard]] bool Write(const void* source, std::size_t size) noexcept;
    [[nodiscard]] std::optional<std::string> ReadLine() noexcept;

    [[nodiscard]] bool Seek(std::int64_t offset,
                            FileSeekOrigin origin = FileSeekOrigin::Begin) noexcept;
    [[nodiscard]] std::optional<std::uint64_t> Position() noexcept;
    [[nodiscard]] std::optional<std::uintmax_t> Size() noexcept;
    [[nodiscard]] bool Flush() noexcept;

    [[nodiscard]] bool Eof() const noexcept;

    [[nodiscard]] std::fstream& Stream() noexcept;
    [[nodiscard]] const std::fstream& Stream() const noexcept;

    [[nodiscard]] static std::optional<std::vector<std::byte>> ReadAllBytes(
        const FilePath& path) noexcept;
    [[nodiscard]] static std::optional<std::string> ReadAllText(const FilePath& path) noexcept;
    [[nodiscard]] static std::optional<std::vector<std::string>> ReadAllLines(
        const FilePath& path) noexcept;

    [[nodiscard]] static bool WriteBytes(const FilePath& path,
                                         Span<const std::byte> content) noexcept;
    [[nodiscard]] static bool WriteText(const FilePath& path, std::string_view content) noexcept;
    [[nodiscard]] static bool AppendBytes(const FilePath& path,
                                          Span<const std::byte> content) noexcept;
    [[nodiscard]] static bool AppendText(const FilePath& path, std::string_view content) noexcept;

    [[nodiscard]] static bool Exists(const FilePath& path) noexcept;
    [[nodiscard]] static bool IsRegularFile(const FilePath& path) noexcept;
    [[nodiscard]] static std::optional<std::uintmax_t> Size(const FilePath& path) noexcept;
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
    [[nodiscard]] bool CanRead() const noexcept;
    [[nodiscard]] bool CanWrite() const noexcept;

    std::fstream m_stream;
    FilePath m_path;
    FileMode m_mode = FileMode::Read;
}; // class File

} // namespace rad
