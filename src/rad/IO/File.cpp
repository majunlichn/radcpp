#include <rad/IO/File.h>

#include <rad/Core/Crc.h>
#include <rad/Core/String.h>
#include <rad/System/OS.h>

#include <boost/hash2/sha2.hpp>
#include <xxhash.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#if defined(RAD_OS_WINDOWS)
#include <io.h>
#include <share.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace rad
{
namespace
{

[[nodiscard]] std::error_code IoError(int code = errno) noexcept
{
    return code != 0 ? std::error_code(code, std::generic_category())
                     : std::make_error_code(std::errc::io_error);
}

[[nodiscard]] std::error_code TakeStreamError(std::FILE* stream) noexcept
{
    const auto error = IoError();
    // stdio error flags persist; don't attribute this error to the next operation.
    std::clearerr(stream);
    return error;
}

[[nodiscard]] bool IsValidPath(const FilePath& path) noexcept
{
    return !path.empty() &&
           path.native().find(FilePath::value_type{}) == FilePath::string_type::npos;
}

[[nodiscard]] constexpr std::optional<int> GetOpenFlags(
    FileAccess access, FileOpenDisposition disposition) noexcept
{
    int flags = 0;
    switch (access)
    {
    case FileAccess::Read:
        if (disposition != FileOpenDisposition::OpenExisting)
        {
            return std::nullopt;
        }
        flags |= O_RDONLY;
        break;
    case FileAccess::Write:
        flags |= O_WRONLY;
        break;
    case FileAccess::ReadWrite:
        flags |= O_RDWR;
        break;
    case FileAccess::Append:
        if (disposition == FileOpenDisposition::CreateAlways ||
            disposition == FileOpenDisposition::TruncateExisting)
        {
            return std::nullopt;
        }
        flags |= O_WRONLY | O_APPEND;
        break;
    default:
        return std::nullopt;
    }

    switch (disposition)
    {
    case FileOpenDisposition::OpenExisting:
        break;
    case FileOpenDisposition::OpenOrCreate:
        flags |= O_CREAT;
        break;
    case FileOpenDisposition::CreateNew:
        flags |= O_CREAT | O_EXCL;
        break;
    case FileOpenDisposition::CreateAlways:
        flags |= O_CREAT | O_TRUNC;
        break;
    case FileOpenDisposition::TruncateExisting:
        flags |= O_TRUNC;
        break;
    default:
        return std::nullopt;
    }
    return flags;
}

[[nodiscard]] Result<int> OpenDescriptor(const FilePath& path, int flags,
                                         bool temporary = false) noexcept
{
#if defined(RAD_OS_WINDOWS)
    int descriptor = -1;
    const auto error = _wsopen_s(&descriptor, path.c_str(), flags | _O_BINARY | _O_NOINHERIT,
                                 temporary ? _SH_DENYRW : _SH_DENYNO, _S_IREAD | _S_IWRITE);
    if (error != 0)
    {
        return Failure(IoError(error));
    }
#else
    int descriptor;
    do
    {
        descriptor = ::open(path.c_str(), flags | O_CLOEXEC, temporary ? 0600 : 0666);
    } while (descriptor < 0 && errno == EINTR);
    if (descriptor < 0)
    {
        return Failure(IoError());
    }
    // POSIX open can succeed on a directory in read-only mode; the Windows CRT rejects it.
    if ((flags & O_ACCMODE) == O_RDONLY)
    {
        struct stat status{};
        if (::fstat(descriptor, &status) != 0)
        {
            const auto error = IoError();
            ::close(descriptor);
            return Failure(error);
        }
        if (S_ISDIR(status.st_mode))
        {
            ::close(descriptor);
            return Failure(std::errc::is_a_directory);
        }
    }
#endif
    return Success(descriptor);
}

[[nodiscard]] constexpr std::string MakeTemporaryFilename(std::string hex,
                                                          FilePath::value_type targetDigit)
{
    const char first = hex.front();
    if (targetDigit == first ||
        (first >= 'a' && first <= 'f' && targetDigit == first - 'a' + 'A'))
    {
        hex.front() = first == '0' ? '1' : '0';
    }
    return "." + std::move(hex) + ".tmp";
}

static_assert(MakeTemporaryFilename("abcdef0123456789abcdef0123456789", 'A') ==
              ".0bcdef0123456789abcdef0123456789.tmp");
static_assert(MakeTemporaryFilename("0123456789abcdef0123456789abcdef", '0') ==
              ".1123456789abcdef0123456789abcdef.tmp");

template <typename F>
[[nodiscard]] bool ProcessChunks(const FilePath& path, F&& process) noexcept
{
    File file;
    if (!file.Open(path, FileAccess::Read))
    {
        return false;
    }

    std::array<std::byte, 64 * 1024> buffer;
    while (true)
    {
        const auto count = file.Read(buffer);
        if (!count || (count.value() != 0 && !process(buffer.data(), count.value())))
        {
            return false;
        }
        if (count.value() < buffer.size())
        {
            return true;
        }
    }
}

template <typename Buffer>
[[nodiscard]] Result<Buffer> ReadWholeFile(const FilePath& path) noexcept
{
    File file;
    if (const auto opened = file.Open(path, FileAccess::Read); !opened)
    {
        return Failure(opened.error());
    }
    const auto size = file.Size();
    if (!size)
    {
        return Failure(size.error());
    }

    try
    {
        Buffer output;
        if (size.value() > output.max_size())
        {
            return Failure(std::errc::value_too_large);
        }
        output.resize(static_cast<std::size_t>(size.value()));
        const auto count = file.Read(std::as_writable_bytes(std::span(output)));
        if (!count)
        {
            return Failure(count.error());
        }
        output.resize(count.value());

        // The size is a snapshot; read any additional bytes until EOF.
        std::array<typename Buffer::value_type, 64 * 1024> extra;
        while (!file.Eof())
        {
            const auto read = file.Read(std::as_writable_bytes(std::span(extra)));
            if (!read)
            {
                return Failure(read.error());
            }
            if (read.value() == 0)
            {
                break;
            }
            output.insert(output.end(), extra.data(), extra.data() + read.value());
        }
        return Success(std::move(output));
    }
    catch (const std::bad_alloc&)
    {
        return Failure(std::errc::not_enough_memory);
    }
    catch (const std::length_error&)
    {
        return Failure(std::errc::value_too_large);
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
    static_cast<void>(Close());
}

File::File(File&& other) noexcept :
    m_stream(std::exchange(other.m_stream, nullptr)),
    m_path(std::move(other.m_path)),
    m_access(other.m_access),
    m_streamDirection(std::exchange(other.m_streamDirection, StreamDirection::None))
{
}

File& File::operator=(File&& other) noexcept
{
    if (this != &other)
    {
        static_cast<void>(Close());
        m_stream = std::exchange(other.m_stream, nullptr);
        m_path = std::move(other.m_path);
        m_access = other.m_access;
        m_streamDirection = std::exchange(other.m_streamDirection, StreamDirection::None);
    }
    return *this;
}

Result<void> File::Open(const FilePath& path, FileAccess access,
                        FileOpenDisposition disposition) noexcept
{
    const auto flags = GetOpenFlags(access, disposition);
    if (!flags || !IsValidPath(path))
    {
        return Failure(std::errc::invalid_argument);
    }

    try
    {
        // Copy before closing: path may refer to this object's Path().
        FilePath ownedPath = path;
        if (auto closed = Close(); !closed)
        {
            return closed;
        }
        auto descriptor = OpenDescriptor(ownedPath, *flags);
        if (!descriptor)
        {
            return Failure(descriptor.error());
        }
        return AdoptDescriptor(descriptor.value(), std::move(ownedPath), access);
    }
    catch (const std::bad_alloc&)
    {
        return Failure(std::errc::not_enough_memory);
    }
}

Result<void> File::AdoptDescriptor(int descriptor, FilePath path, FileAccess access) noexcept
{
    // fdopen adopts the descriptor without reopening or truncating the path.
    const char* mode = "wb";
    switch (access)
    {
    case FileAccess::Read:
        mode = "rb";
        break;
    case FileAccess::ReadWrite:
        mode = "r+b";
        break;
    case FileAccess::Append:
        mode = "ab";
        break;
    case FileAccess::Write:
        break;
    }
    errno = 0;
#if defined(RAD_OS_WINDOWS)
    m_stream = _fdopen(descriptor, mode);
#else
    m_stream = ::fdopen(descriptor, mode);
#endif
    if (m_stream == nullptr)
    {
        const auto error = IoError();
#if defined(RAD_OS_WINDOWS)
        _close(descriptor);
#else
        ::close(descriptor);
#endif
        return Failure(error);
    }
    m_path = std::move(path);
    m_access = access;
    return Success();
}

Result<void> File::Close() noexcept
{
    auto* stream = std::exchange(m_stream, nullptr);
    if (stream == nullptr)
    {
        return Success();
    }
    m_streamDirection = StreamDirection::None;
    errno = 0;
    if (std::fclose(stream) != 0)
    {
        return Failure(IoError());
    }
    return Success();
}

bool File::IsOpen() const noexcept
{
    return m_stream != nullptr;
}

const FilePath& File::Path() const noexcept
{
    return m_path;
}

FileAccess File::Access() const noexcept
{
    return m_access;
}

Result<void> File::PrepareRead(bool transferBytes) noexcept
{
    if (!IsOpen() || (m_access != FileAccess::Read && m_access != FileAccess::ReadWrite))
    {
        return Failure(std::errc::bad_file_descriptor);
    }
    if (transferBytes && m_streamDirection == StreamDirection::Write)
    {
        return Flush();
    }
    return Success();
}

Result<void> File::PrepareWrite(bool transferBytes) noexcept
{
    if (!IsOpen() || m_access == FileAccess::Read)
    {
        return Failure(std::errc::bad_file_descriptor);
    }
    if (transferBytes && m_streamDirection == StreamDirection::Read && !Eof())
    {
        // Update streams require positioning after input, unless it reached EOF.
        return Seek(0, FileSeekOrigin::Current);
    }
    return Success();
}

Result<std::size_t> File::Read(Span<std::byte> destination) noexcept
{
    if (auto prepared = PrepareRead(!destination.empty()); !prepared)
    {
        return Failure(prepared.error());
    }
    if (destination.empty())
    {
        return Success(std::size_t{0});
    }
    m_streamDirection = StreamDirection::Read;
    errno = 0;
    const auto count = std::fread(destination.data(), 1, destination.size(), m_stream);
    if (count < destination.size() && std::ferror(m_stream))
    {
        return Failure(TakeStreamError(m_stream));
    }
    return Success(count);
}

Result<std::size_t> File::Read(void* destination, std::size_t size) noexcept
{
    if (destination == nullptr && size != 0)
    {
        return Failure(std::errc::invalid_argument);
    }
    return Read({static_cast<std::byte*>(destination), size});
}

Result<void> File::Write(Span<const std::byte> source) noexcept
{
    if (auto prepared = PrepareWrite(!source.empty()); !prepared)
    {
        return prepared;
    }
    if (!source.empty())
    {
        const bool atEof = m_streamDirection == StreamDirection::Read && Eof();
        m_streamDirection = StreamDirection::Write;
        errno = 0;
        const auto count = std::fwrite(source.data(), 1, source.size(), m_stream);
        if (count != source.size() || std::ferror(m_stream))
        {
            return Failure(TakeStreamError(m_stream));
        }
        // The CRT may need the EOF flag for the transition, so clear it only after writing.
        if (atEof)
        {
            std::clearerr(m_stream);
        }
    }
    return Success();
}

Result<void> File::Write(const void* source, std::size_t size) noexcept
{
    if (source == nullptr && size != 0)
    {
        return Failure(std::errc::invalid_argument);
    }
    return Write({static_cast<const std::byte*>(source), size});
}

bool File::ReadLine(std::string& line, std::error_code& ec) noexcept
{
    line.clear();
    ec.clear();
    if (auto prepared = PrepareRead(); !prepared)
    {
        ec = prepared.error();
        return false;
    }

    try
    {
        m_streamDirection = StreamDirection::Read;
        errno = 0;
        for (;;)
        {
            // The stream is private; avoid locking it for every byte of a line.
#if defined(RAD_OS_WINDOWS)
            const auto character = _fgetc_nolock(m_stream);
#else
            const auto character = ::getc_unlocked(m_stream);
#endif
            if (character == EOF)
            {
                if (std::ferror(m_stream))
                {
                    ec = TakeStreamError(m_stream);
                    line.clear();
                    return false;
                }
                if (line.empty())
                {
                    return false;
                }
                break;
            }
            if (character == '\n')
            {
                break;
            }
            line.push_back(static_cast<char>(character));
        }
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        return true;
    }
    catch (const std::bad_alloc&)
    {
        ec = std::make_error_code(std::errc::not_enough_memory);
    }
    catch (const std::length_error&)
    {
        ec = std::make_error_code(std::errc::value_too_large);
    }
    line.clear();
    return false;
}

bool File::ReadLine(std::string& line)
{
    std::error_code ec;
    const bool read = ReadLine(line, ec);
    if (ec)
    {
        throw std::system_error(ec, "File::ReadLine");
    }
    return read;
}

Result<void> File::Seek(std::int64_t offset, FileSeekOrigin origin) noexcept
{
    if (!IsOpen())
    {
        return Failure(std::errc::bad_file_descriptor);
    }
    int whence;
    switch (origin)
    {
    case FileSeekOrigin::Begin:
        whence = SEEK_SET;
        break;
    case FileSeekOrigin::Current:
        whence = SEEK_CUR;
        break;
    case FileSeekOrigin::End:
        whence = SEEK_END;
        break;
    default:
        return Failure(std::errc::invalid_argument);
    }
    // Some CRTs do not propagate a flush failure from the seek itself.
    if (auto flushed = Flush(); !flushed)
    {
        return flushed;
    }
    errno = 0;
#if defined(RAD_OS_WINDOWS)
    const auto result = _fseeki64(m_stream, offset, whence);
#else
    static_assert(sizeof(off_t) >= sizeof(offset));
    const auto result = ::fseeko(m_stream, static_cast<off_t>(offset), whence);
#endif
    if (result != 0)
    {
        return Failure(IoError());
    }
    std::clearerr(m_stream);
    m_streamDirection = StreamDirection::None;
    return Success();
}

Result<std::uint64_t> File::Position() noexcept
{
    if (!IsOpen())
    {
        return Failure(std::errc::bad_file_descriptor);
    }
    errno = 0;
#if defined(RAD_OS_WINDOWS)
    const auto position = _ftelli64(m_stream);
#else
    const auto position = ::ftello(m_stream);
#endif
    if (position < 0)
    {
        return Failure(IoError());
    }
    return Success(static_cast<std::uint64_t>(position));
}

Result<std::uint64_t> File::Size() noexcept
{
    if (auto flushed = Flush(); !flushed)
    {
        return Failure(flushed.error());
    }

    // Query the opened file, not a path that may now refer to a replacement.
    errno = 0;
#if defined(RAD_OS_WINDOWS)
    struct _stat64 status{};
    if (_fstat64(_fileno(m_stream), &status) != 0)
    {
        return Failure(IoError());
    }
    const bool regular = (status.st_mode & _S_IFMT) == _S_IFREG;
#else
    struct stat status{};
    if (::fstat(::fileno(m_stream), &status) != 0)
    {
        return Failure(IoError());
    }
    const bool regular = S_ISREG(status.st_mode);
#endif
    if (!regular)
    {
        return Failure(std::errc::operation_not_supported);
    }
    if (!std::in_range<std::uint64_t>(status.st_size))
    {
        return Failure(std::errc::value_too_large);
    }
    return Success(static_cast<std::uint64_t>(status.st_size));
}

Result<void> File::Flush() noexcept
{
    if (!IsOpen())
    {
        return Failure(std::errc::bad_file_descriptor);
    }
    if (m_streamDirection == StreamDirection::Write)
    {
        errno = 0;
        if (std::fflush(m_stream) != 0)
        {
            return Failure(TakeStreamError(m_stream));
        }
        m_streamDirection = StreamDirection::None;
    }
    return Success();
}

bool File::Eof() const noexcept
{
    return IsOpen() && std::feof(m_stream);
}

Result<std::vector<std::byte>> File::ReadAllBytes(const FilePath& path) noexcept
{
    return ReadWholeFile<std::vector<std::byte>>(path);
}

Result<std::string> File::ReadAllText(const FilePath& path) noexcept
{
    return ReadWholeFile<std::string>(path);
}

Result<std::vector<std::string>> File::ReadAllLines(const FilePath& path) noexcept
{
    File file;
    if (const auto opened = file.Open(path, FileAccess::Read); !opened)
    {
        return Failure(opened.error());
    }
    try
    {
        std::vector<std::string> lines;
        std::string line;
        std::error_code ec;
        while (file.ReadLine(line, ec))
        {
            lines.push_back(std::move(line));
        }
        if (ec)
        {
            return Failure(ec);
        }
        return Success(std::move(lines));
    }
    catch (const std::bad_alloc&)
    {
        return Failure(std::errc::not_enough_memory);
    }
    catch (const std::length_error&)
    {
        return Failure(std::errc::value_too_large);
    }
}

Result<void> File::WriteBytes(const FilePath& path, Span<const std::byte> content) noexcept
{
    if (!IsValidPath(path) || !path.has_filename())
    {
        return Failure(std::errc::invalid_argument);
    }

    try
    {
        std::error_code error;
        const auto destination = std::filesystem::absolute(path, error);
        if (error)
        {
            return Failure(error);
        }
#if defined(RAD_OS_WINDOWS)
        if (destination.filename().native().find(L':') != FilePath::string_type::npos)
        {
            return Failure(std::errc::invalid_argument);
        }
#endif

        const auto status = std::filesystem::symlink_status(destination, error);
        if (error && error != std::errc::no_such_file_or_directory)
        {
            return Failure(error);
        }
        if (status.type() != FileType::not_found && !std::filesystem::is_regular_file(status))
        {
            return Failure(std::filesystem::is_directory(status) ? std::errc::is_a_directory
                                                                 : std::errc::invalid_argument);
        }

        constexpr int flags =
            GetOpenFlags(FileAccess::Write, FileOpenDisposition::CreateNew).value();
        const auto parent = destination.parent_path();
        const auto filename = destination.filename();
        const auto& name = filename.native();
        // A missing destination must not be used as the temporary file, even on Windows.
        const auto targetDigit =
            name.size() > 1 && name[0] == '.' ? name[1] : FilePath::value_type{};

        for (int attempt = 0; attempt < 8; ++attempt)
        {
            const auto candidate =
                parent / MakeTemporaryFilename(ToHexString(os::urandom(16)), targetDigit);
            const auto descriptor = OpenDescriptor(candidate, flags, true);
            if (!descriptor)
            {
                if (descriptor.error() == std::errc::file_exists)
                {
                    continue;
                }
                return Failure(descriptor.error());
            }

            // Only clean up a path we created. On failure, close the file before deleting it.
            struct Cleaner
            {
                const FilePath* m_path;

                ~Cleaner() noexcept
                {
                    if (m_path)
                    {
#if defined(RAD_OS_WINDOWS)
                        DeleteFileW(m_path->c_str());
#else
                        ::unlink(m_path->c_str());
#endif
                    }
                }
            } cleaner{&candidate};

            File file;
            if (auto adopted = file.AdoptDescriptor(descriptor.value(), {}, FileAccess::Write);
                !adopted)
            {
                return adopted;
            }
            if (auto written = file.Write(content); !written)
            {
                return written;
            }
            if (auto closed = file.Close(); !closed)
            {
                return closed;
            }

            os::replace(candidate, destination);
            cleaner.m_path = nullptr;
            return Success();
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

Result<void> File::WriteText(const FilePath& path, std::string_view content) noexcept
{
    return WriteBytes(path, std::as_bytes(std::span(content)));
}

Result<void> File::AppendBytes(const FilePath& path, Span<const std::byte> content) noexcept
{
    File file;
    if (auto opened = file.Open(path, FileAccess::Append, FileOpenDisposition::OpenOrCreate);
        !opened)
    {
        return opened;
    }
    if (auto written = file.Write(content); !written)
    {
        return written;
    }
    return file.Close();
}

Result<void> File::AppendText(const FilePath& path, std::string_view content) noexcept
{
    return AppendBytes(path, std::as_bytes(std::span(content)));
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

Result<std::uint64_t> File::Size(const FilePath& path) noexcept
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

} // namespace rad
