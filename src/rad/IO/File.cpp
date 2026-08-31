#include <rad/IO/File.h>

#include <rad/Core/Crc.h>

#include <boost/hash2/sha2.hpp>
#include <xxhash.h>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace rad
{
namespace
{

[[nodiscard]] std::ios::openmode ToOpenMode(FileMode mode) noexcept
{
    std::ios::openmode result = std::ios::binary;
    switch (mode)
    {
    case FileMode::Read:
        result |= std::ios::in;
        break;
    case FileMode::Write:
        result |= std::ios::out | std::ios::trunc;
        break;
    case FileMode::ReadWrite:
        result |= std::ios::in | std::ios::out;
        break;
    case FileMode::Append:
        result |= std::ios::out | std::ios::app;
        break;
    }

    return result;
}

[[nodiscard]] std::ios::seekdir ToSeekDirection(FileSeekOrigin origin) noexcept
{
    switch (origin)
    {
    case FileSeekOrigin::Begin:
        return std::ios::beg;
    case FileSeekOrigin::Current:
        return std::ios::cur;
    case FileSeekOrigin::End:
        return std::ios::end;
    }
    return std::ios::beg;
}

template <typename F>
[[nodiscard]] bool ProcessChunks(const FilePath& path, F&& process) noexcept
{
    File file;
    if (!file.Open(path, FileMode::Read))
    {
        return false;
    }

    std::array<std::byte, 64 * 1024> buffer{};
    while (true)
    {
        const auto size = file.Read(buffer);
        if (file.Stream().bad() || (file.Stream().fail() && !file.Eof()))
        {
            return false;
        }
        if (size != 0 && !process(buffer.data(), size))
        {
            return false;
        }
        if (size < buffer.size())
        {
            return true;
        }
    }
}

struct XxHash64StateDeleter
{
    void operator()(XXH64_state_t* state) const noexcept
    {
        static_cast<void>(XXH64_freeState(state));
    }
};

} // namespace

File::~File() noexcept
{
    if (m_stream.is_open())
    {
        m_stream.close();
    }
}

File::File(File&& other) noexcept :
    m_stream(std::move(other.m_stream)),
    m_path(std::move(other.m_path)),
    m_mode(other.m_mode)
{
    other.m_path.clear();
}

File& File::operator=(File&& other) noexcept
{
    if (this == &other)
    {
        return *this;
    }

    if (m_stream.is_open())
    {
        m_stream.close();
    }
    m_stream = std::move(other.m_stream);
    m_path = std::move(other.m_path);
    m_mode = other.m_mode;

    other.m_path.clear();
    return *this;
}

bool File::Open(const FilePath& path, FileMode mode) noexcept
{
    if (m_stream.is_open() && !Close())
    {
        return false;
    }

    m_stream.clear();
    try
    {
        m_path = path;
        m_mode = mode;
        m_stream.open(m_path, ToOpenMode(m_mode));
    }
    catch (...)
    {
        return false;
    }

    if (!m_stream.is_open())
    {
        m_stream.clear();
        return false;
    }
    return true;
}

bool File::Close() noexcept
{
    if (!m_stream.is_open())
    {
        return true;
    }

    const bool hadError = m_stream.bad() || (m_stream.fail() && !m_stream.eof());
    m_stream.clear();
    m_stream.close();
    if (hadError || m_stream.fail())
    {
        m_stream.clear();
        return false;
    }
    m_stream.clear();
    return true;
}

bool File::IsOpen() const noexcept
{
    return m_stream.is_open();
}

File::operator bool() const noexcept
{
    return m_stream.is_open() && !m_stream.fail();
}

const FilePath& File::Path() const noexcept
{
    return m_path;
}

FileMode File::Mode() const noexcept
{
    return m_mode;
}

std::size_t File::Read(Span<std::byte> destination) noexcept
{
    if (!m_stream.is_open())
    {
        return 0;
    }
    if (!CanRead())
    {
        return 0;
    }
    if (destination.empty())
    {
        return 0;
    }

    std::size_t total = 0;
    constexpr auto maxChunk = static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max());
    while (total < destination.size())
    {
        const auto chunkSize = std::min(destination.size() - total, maxChunk);
        m_stream.read(reinterpret_cast<char*>(destination.data() + total),
                      static_cast<std::streamsize>(chunkSize));
        const auto count = m_stream.gcount();
        if (count > 0)
        {
            total += static_cast<std::size_t>(count);
        }

        if (m_stream.bad() || (m_stream.fail() && !m_stream.eof()))
        {
            break;
        }
        if (m_stream.eof() || static_cast<std::size_t>(count) < chunkSize)
        {
            break;
        }
    }
    return total;
}

std::size_t File::Read(void* destination, std::size_t size) noexcept
{
    if (destination == nullptr && size != 0)
    {
        return 0;
    }
    return Read({static_cast<std::byte*>(destination), size});
}

bool File::Write(Span<const std::byte> source) noexcept
{
    if (!m_stream.is_open())
    {
        return false;
    }
    if (!CanWrite())
    {
        return false;
    }

    std::size_t total = 0;
    constexpr auto maxChunk = static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max());
    while (total < source.size())
    {
        const auto chunkSize = std::min(source.size() - total, maxChunk);
        m_stream.write(reinterpret_cast<const char*>(source.data() + total),
                       static_cast<std::streamsize>(chunkSize));
        if (!m_stream)
        {
            return false;
        }
        total += chunkSize;
    }
    return true;
}

bool File::Write(const void* source, std::size_t size) noexcept
{
    if (source == nullptr && size != 0)
    {
        return false;
    }
    return Write({static_cast<const std::byte*>(source), size});
}

std::optional<std::string> File::ReadLine() noexcept
{
    if (!m_stream.is_open())
    {
        return std::nullopt;
    }
    if (!CanRead())
    {
        return std::nullopt;
    }

    try
    {
        std::string line;
        if (std::getline(m_stream, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            return std::optional<std::string>{std::move(line)};
        }
    }
    catch (...)
    {
        return std::nullopt;
    }
    return std::nullopt;
}

bool File::Seek(std::int64_t offset, FileSeekOrigin origin) noexcept
{
    if (!m_stream.is_open())
    {
        return false;
    }

    m_stream.clear();
    const auto direction = ToSeekDirection(origin);
    if (CanRead())
    {
        m_stream.seekg(static_cast<std::streamoff>(offset), direction);
        if (!m_stream)
        {
            return false;
        }
        if (CanWrite())
        {
            const auto position = m_stream.tellg();
            m_stream.seekp(position);
        }
    }
    else
    {
        m_stream.seekp(static_cast<std::streamoff>(offset), direction);
    }

    if (!m_stream)
    {
        return false;
    }
    return true;
}

std::optional<std::uint64_t> File::Position() noexcept
{
    if (!m_stream.is_open())
    {
        return std::nullopt;
    }

    const auto oldState = m_stream.rdstate();
    m_stream.clear();
    auto position = CanRead() ? m_stream.tellg() : m_stream.tellp();
    if (position == std::streampos{-1} && CanWrite())
    {
        m_stream.clear();
        position = m_stream.tellp();
    }
    m_stream.clear(oldState);

    if (position == std::streampos{-1})
    {
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(position);
}

std::optional<std::uintmax_t> File::Size() noexcept
{
    if (!m_stream.is_open())
    {
        return std::nullopt;
    }
    if (CanWrite() && !Flush())
    {
        return std::nullopt;
    }

    std::error_code error;
    const auto size = std::filesystem::file_size(m_path, error);
    if (error)
    {
        return std::nullopt;
    }
    return size;
}

bool File::Flush() noexcept
{
    if (!m_stream.is_open())
    {
        return false;
    }
    if (!CanWrite())
    {
        return true;
    }

    m_stream.flush();
    if (!m_stream)
    {
        return false;
    }
    return true;
}

bool File::Eof() const noexcept
{
    return m_stream.eof();
}

std::fstream& File::Stream() noexcept
{
    return m_stream;
}

const std::fstream& File::Stream() const noexcept
{
    return m_stream;
}

std::optional<std::vector<std::byte>> File::ReadAllBytes(const FilePath& path) noexcept
{
    File file;
    if (!file.Open(path, FileMode::Read))
    {
        return std::nullopt;
    }

    const auto fileSize = file.Size();
    if (!fileSize || *fileSize > std::numeric_limits<std::size_t>::max())
    {
        return std::nullopt;
    }

    std::optional<std::vector<std::byte>> result;
    try
    {
        result.emplace(static_cast<std::size_t>(*fileSize));
    }
    catch (const std::bad_alloc&)
    {
        return std::nullopt;
    }
    catch (const std::length_error&)
    {
        return std::nullopt;
    }

    auto& bytes = *result;
    const auto count = file.Read(bytes);
    if (file.Stream().bad() || (file.Stream().fail() && !file.Eof()))
    {
        return std::nullopt;
    }
    bytes.resize(count);

    std::array<std::byte, 64 * 1024> buffer{};
    while (!file.Eof())
    {
        const auto extra = file.Read(buffer);
        if (file.Stream().bad() || (file.Stream().fail() && !file.Eof()))
        {
            return std::nullopt;
        }
        if (extra == 0)
        {
            break;
        }
        try
        {
            bytes.insert(bytes.end(), buffer.begin(), buffer.begin() + extra);
        }
        catch (const std::bad_alloc&)
        {
            return std::nullopt;
        }
        catch (const std::length_error&)
        {
            return std::nullopt;
        }
    }
    return result;
}

std::optional<std::string> File::ReadAllText(const FilePath& path) noexcept
{
    auto bytes = ReadAllBytes(path);
    if (!bytes)
    {
        return std::nullopt;
    }
    if (bytes->empty())
    {
        return std::string{};
    }

    try
    {
        return std::string{reinterpret_cast<const char*>(bytes->data()), bytes->size()};
    }
    catch (const std::bad_alloc&)
    {
        return std::nullopt;
    }
    catch (const std::length_error&)
    {
        return std::nullopt;
    }
}

std::optional<std::vector<std::string>> File::ReadAllLines(const FilePath& path) noexcept
{
    File file;
    if (!file.Open(path, FileMode::Read))
    {
        return std::nullopt;
    }

    try
    {
        std::vector<std::string> result;
        while (auto line = file.ReadLine())
        {
            result.push_back(std::move(*line));
        }
        if (!file.Eof() || file.Stream().bad())
        {
            return std::nullopt;
        }
        return result;
    }
    catch (const std::bad_alloc&)
    {
        return std::nullopt;
    }
    catch (const std::length_error&)
    {
        return std::nullopt;
    }
}

bool File::WriteBytes(const FilePath& path, Span<const std::byte> content) noexcept
{
    File file;
    return file.Open(path, FileMode::Write) && file.Write(content) && file.Close();
}

bool File::WriteText(const FilePath& path, std::string_view content) noexcept
{
    return WriteBytes(path, {reinterpret_cast<const std::byte*>(content.data()), content.size()});
}

bool File::AppendBytes(const FilePath& path, Span<const std::byte> content) noexcept
{
    File file;
    return file.Open(path, FileMode::Append) && file.Write(content) && file.Close();
}

bool File::AppendText(const FilePath& path, std::string_view content) noexcept
{
    return AppendBytes(path, {reinterpret_cast<const std::byte*>(content.data()), content.size()});
}

bool File::Exists(const FilePath& path) noexcept
{
    std::error_code error;
    return std::filesystem::exists(path, error) && !error;
}

bool File::IsRegularFile(const FilePath& path) noexcept
{
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error;
}

std::optional<std::uintmax_t> File::Size(const FilePath& path) noexcept
{
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    return error ? std::nullopt : std::optional<std::uintmax_t>{size};
}

std::optional<FileTime> File::LastWriteTime(const FilePath& path) noexcept
{
    std::error_code error;
    const auto time = std::filesystem::last_write_time(path, error);
    return error ? std::nullopt : std::optional{time};
}

std::optional<std::uint32_t> File::Crc32(const FilePath& path) noexcept
{
    rad::Crc32 hash;
    if (!ProcessChunks(path,
                       [&](const std::byte* data, std::size_t size)
                       {
                           hash.Update(data, size);
                           return true;
                       }))
    {
        return std::nullopt;
    }
    return hash.Value();
}

std::optional<std::uint64_t> File::XxHash64(const FilePath& path) noexcept
{
    std::unique_ptr<XXH64_state_t, XxHash64StateDeleter> state{XXH64_createState()};
    if (!state || XXH64_reset(state.get(), 0) != XXH_OK)
    {
        return std::nullopt;
    }

    if (!ProcessChunks(path, [&](const std::byte* data, std::size_t size)
                       { return XXH64_update(state.get(), data, size) == XXH_OK; }))
    {
        return std::nullopt;
    }
    return XXH64_digest(state.get());
}

std::optional<Sha256Digest> File::Sha256(const FilePath& path) noexcept
{
    boost::hash2::sha2_256 hash;
    if (!ProcessChunks(path,
                       [&](const std::byte* data, std::size_t size)
                       {
                           hash.update(data, size);
                           return true;
                       }))
    {
        return std::nullopt;
    }

    const auto digest = hash.result();
    Sha256Digest result{};
    std::transform(digest.begin(), digest.end(), result.begin(),
                   [](unsigned char value) { return static_cast<std::byte>(value); });
    return result;
}

bool File::Remove(const FilePath& path) noexcept
{
    std::error_code error;
    return std::filesystem::remove(path, error) && !error;
}

bool File::Copy(const FilePath& source, const FilePath& destination, bool overwrite) noexcept
{
    const auto options = overwrite ? std::filesystem::copy_options::overwrite_existing
                                   : std::filesystem::copy_options::none;
    std::error_code error;
    return std::filesystem::copy_file(source, destination, options, error) && !error;
}

bool File::Move(const FilePath& source, const FilePath& destination) noexcept
{
    std::error_code error;
    std::filesystem::rename(source, destination, error);
    return !error;
}

bool File::CanRead() const noexcept
{
    return (m_mode == FileMode::Read) || (m_mode == FileMode::ReadWrite);
}

bool File::CanWrite() const noexcept
{
    return (m_mode == FileMode::Write) || (m_mode == FileMode::ReadWrite) ||
           (m_mode == FileMode::Append);
}

} // namespace rad
