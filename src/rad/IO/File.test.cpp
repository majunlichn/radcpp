#include <rad/IO/File.h>

#include <rad/Core/String.h>
#include <rad/System/FileSystem.h>
#include <rad/System/OS.h>

#include <gtest/gtest.h>

#include <iterator>
#include <utility>
#include <vector>

namespace
{

struct TestDirectory
{
    rad::FilePath path = "rad-file-test-" + rad::ToHexString(rad::os::urandom(16));
    bool created = std::filesystem::create_directory(path);

    ~TestDirectory()
    {
        if (created)
        {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    }
};

void ExpectText(const rad::FilePath& path, std::string_view expected)
{
    const auto text = rad::File::ReadAllText(path);
    ASSERT_TRUE(text) << text.error().message();
    EXPECT_EQ(text.value(), expected);
}

} // namespace

TEST(File, CreateTemporary)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    const auto parent = std::filesystem::absolute(directory.path);

    // The file supports shared I/O and remains on disk after the handles close.
    {
        rad::FilePath path;
        {
            auto created = rad::File::CreateTemporary(directory.path);
            ASSERT_TRUE(created) << created.error().message();
            auto& file = created.value();
            path = file.Path();
            EXPECT_EQ(path.parent_path(), parent);
#if !defined(RAD_OS_WINDOWS)
            using std::filesystem::perms;
            EXPECT_EQ(std::filesystem::status(path).permissions() & ~perms::owner_all, perms::none);
#endif
            ASSERT_TRUE(file.Write("test", 4));
            ASSERT_TRUE(file.Flush());
            ExpectText(path, "test");
            ASSERT_TRUE(file.Seek(0));
            char text[4]{};
            const auto read = file.Read(text, sizeof(text));
            ASSERT_TRUE(read) << read.error().message();
            EXPECT_EQ(read.value(), sizeof(text));
            EXPECT_EQ(std::string_view(text, sizeof(text)), "test");
            auto another = rad::File::CreateTemporary(directory.path);
            ASSERT_TRUE(another) << another.error().message();
            EXPECT_NE(another.value().Path(), path);
        }
        ExpectText(path, "test");
    }

    // Omitting the parent uses the OS temporary directory.
    {
        const auto systemParent = rad::os::temp_directory_path();
        auto file = rad::File::CreateTemporary();
        ASSERT_TRUE(file) << file.error().message();
        const auto path = file.value().Path();
        EXPECT_TRUE(std::filesystem::equivalent(path.parent_path(), systemParent));
        EXPECT_TRUE(file.value().Close());
        EXPECT_TRUE(std::filesystem::remove(path));
    }

    // Creation does not create missing parents or accept a path truncated by the OS.
    {
        const auto missing = directory.path / "missing";
        const auto file = rad::File::CreateTemporary(missing);
        ASSERT_FALSE(file);
        EXPECT_EQ(file.error(), std::errc::no_such_file_or_directory);
        EXPECT_FALSE(std::filesystem::exists(missing));

        auto invalid = directory.path.native();
        invalid.push_back(rad::FilePath::value_type{});
        const auto invalidFile = rad::File::CreateTemporary(invalid);
        ASSERT_FALSE(invalidFile);
        EXPECT_EQ(invalidFile.error(), std::errc::invalid_argument);
    }
}

TEST(File, HashFunctions)
{
    // Test hash functions.
    {
        const TestDirectory directory;
        ASSERT_TRUE(directory.created);
        const auto path = directory.path / "hash.bin";
        ASSERT_TRUE(rad::File::WriteText(path, "123456789"));
        EXPECT_EQ(rad::File::Crc32(path), 0xCBF43926U);
        EXPECT_EQ(rad::File::XxHash64(path), 0x8CB841DB40E6AE83ULL);
        const auto sha256 = rad::File::Sha256(path);
        ASSERT_TRUE(sha256);
        EXPECT_EQ(rad::ToHexString(*sha256),
                  "15e2b0d3c33891ebb0f1ef609ec419420c20e320ce94c65fbc8c3312448eb225");
    }
}

TEST(File, WholeFileWriteReplacesDestination)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    const auto path = directory.path / "target.bin";
    const auto link = directory.path / "original.bin";
    ASSERT_TRUE(rad::File::WriteText(path, "old content"));
    std::filesystem::create_hard_link(path, link);

    // Stream writes still truncate the existing file in place, affecting both names.
    rad::File file;
    ASSERT_TRUE(file.Open(link, rad::FileAccess::Write, rad::FileOpenDisposition::CreateAlways));
    ASSERT_TRUE(file.Write("old", 3));
    ASSERT_TRUE(file.Close());
    ASSERT_NO_FATAL_FAILURE(ExpectText(path, "old"));

    // The destination is replaced, while another name for the original file is untouched.
    const std::vector<std::byte> bytes{std::byte{0}, std::byte{0xFF}};
    ASSERT_TRUE(rad::File::WriteBytes(path, bytes));
    const auto read = rad::File::ReadAllBytes(path);
    ASSERT_TRUE(read) << read.error().message();
    EXPECT_EQ(read.value(), bytes);
    ExpectText(link, "old");

    ASSERT_TRUE(rad::File::WriteText(path, ""));
    ExpectText(path, "");
    ExpectText(link, "old");

    // A name matching the temporary-file format is still a valid destination.
    {
        const auto temporaryLikePath = directory.path / ".ABCDEF0123456789ABCDEF0123456789.tmp";
        ASSERT_TRUE(rad::File::WriteText(temporaryLikePath, "new"));
        ExpectText(temporaryLikePath, "new");
    }
}

TEST(File, WholeFileWriteRejectsDirectory)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    const auto path = directory.path / "directory";
    ASSERT_TRUE(std::filesystem::create_directory(path));
    const auto directoryWrite = rad::File::WriteText(path, "new content");
    ASSERT_FALSE(directoryWrite);
    EXPECT_EQ(directoryWrite.error(), std::errc::is_a_directory);
    EXPECT_TRUE(std::filesystem::is_directory(path));
}

#if defined(RAD_OS_WINDOWS)
TEST(File, WholeFileWriteLeavesLockedDestinationUntouched)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    const auto filePath = directory.path / "locked.txt";
    ASSERT_TRUE(rad::File::WriteText(filePath, "old content"));
    rad::File locked;
    ASSERT_TRUE(locked.Open(filePath, rad::FileAccess::Read));
    const auto lockedWrite = rad::File::WriteText(filePath, "new content");
    ASSERT_FALSE(lockedWrite);
    EXPECT_EQ(lockedWrite.error().category(), std::system_category());
    ExpectText(filePath, "old content");
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory.path),
                            std::filesystem::directory_iterator{}),
              1);
}
#else
TEST(File, WholeFileWritePreservesOpenHandle)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    const auto filePath = directory.path / "open.txt";
    ASSERT_TRUE(rad::File::WriteText(filePath, "old"));
    rad::File opened;
    ASSERT_TRUE(opened.Open(filePath, rad::FileAccess::Read));
    ASSERT_TRUE(rad::File::WriteText(filePath, "replacement"));
    const auto openedSize = opened.Size();
    ASSERT_TRUE(openedSize);
    EXPECT_EQ(openedSize.value(), 3u);
    const auto pathSize = rad::GetSize(filePath);
    ASSERT_TRUE(pathSize);
    EXPECT_EQ(pathSize.value(), 11u);
    std::string line;
    ASSERT_TRUE(opened.ReadLine(line));
    EXPECT_EQ(line, "old");
    ExpectText(filePath, "replacement");
}
#endif

TEST(File, OpenDispositionsAndErrors)
{
    using rad::FileAccess;
    using rad::FileOpenDisposition;
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    rad::File file;

    // Neither the default nor TruncateExisting creates a missing file.
    {
        const auto missing = file.Open(directory.path / "missing.bin", FileAccess::Write);
        ASSERT_FALSE(missing);
        EXPECT_EQ(missing.error(), std::errc::no_such_file_or_directory);
        const auto truncateMissing = file.Open(directory.path / "truncate-missing.bin",
                                              FileAccess::Write,
                                              FileOpenDisposition::TruncateExisting);
        ASSERT_FALSE(truncateMissing);
        EXPECT_EQ(truncateMissing.error(), std::errc::no_such_file_or_directory);
    }

    // CreateNew creates a missing file but never overwrites an existing file.
    {
        const auto path = directory.path / "create-new.bin";
        ASSERT_TRUE(file.Open(path, FileAccess::Write, FileOpenDisposition::CreateNew));
        ASSERT_TRUE(file.Write("old", 3));
        ASSERT_TRUE(file.Close());
        const auto collision = file.Open(path, FileAccess::Write, FileOpenDisposition::CreateNew);
        ASSERT_FALSE(collision);
        EXPECT_EQ(collision.error(), std::errc::file_exists);
        ExpectText(path, "old");
    }

    // The default write disposition opens an existing file without truncating it.
    {
        const auto path = directory.path / "open-existing.bin";
        ASSERT_TRUE(rad::File::WriteText(path, "abc"));
        ASSERT_TRUE(file.Open(path, FileAccess::Write));
        ASSERT_TRUE(file.Write("X", 1));
        ASSERT_TRUE(file.Close());
        ExpectText(path, "Xbc");
    }

    // OpenOrCreate creates a missing file and preserves existing bytes when reopened.
    {
        const auto path = directory.path / "open-or-create.bin";
        ASSERT_TRUE(file.Open(path, FileAccess::Write, FileOpenDisposition::OpenOrCreate));
        ASSERT_TRUE(file.Write("abc", 3));
        ASSERT_TRUE(file.Close());
        ASSERT_NO_FATAL_FAILURE(ExpectText(path, "abc"));
        ASSERT_TRUE(file.Open(path, FileAccess::Write, FileOpenDisposition::OpenOrCreate));
        ASSERT_TRUE(file.Write("X", 1));
        ASSERT_TRUE(file.Close());
        ExpectText(path, "Xbc");
    }

    // Append writes at EOF even after an explicit seek.
    {
        const auto path = directory.path / "append.bin";
        ASSERT_TRUE(rad::File::WriteText(path, "abc"));
        ASSERT_TRUE(file.Open(path, FileAccess::Append));
        ASSERT_TRUE(file.Seek(0));
        ASSERT_TRUE(file.Write("!", 1));
        ASSERT_TRUE(file.Close());
        ExpectText(path, "abc!");
    }

    // Opening with TruncateExisting truncates even without a subsequent write.
    {
        const auto path = directory.path / "truncate.bin";
        ASSERT_TRUE(rad::File::WriteText(path, "old"));
        ASSERT_TRUE(file.Open(path, FileAccess::Write, FileOpenDisposition::TruncateExisting));
        ASSERT_TRUE(file.Close());
        ExpectText(path, "");
    }
}

TEST(File, ReadAndWriteRespectAccess)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    const auto path = directory.path / "access.bin";
    ASSERT_TRUE(rad::File::WriteText(path, "old"));
    rad::File file;

    // Empty requests still require an open file and the matching access.
    const auto closed = file.Read(nullptr, 0);
    ASSERT_FALSE(closed);
    EXPECT_EQ(closed.error(), std::errc::bad_file_descriptor);

    ASSERT_TRUE(file.Open(path, rad::FileAccess::Read));
    for (const auto& write : {file.Write("X", 1), file.Write(nullptr, 0)})
    {
        ASSERT_FALSE(write);
        EXPECT_EQ(write.error(), std::errc::bad_file_descriptor);
    }
    ASSERT_TRUE(file.Close());

    ASSERT_TRUE(file.Open(path, rad::FileAccess::Write));
    char byte{};
    for (const auto& read : {file.Read(&byte, 1), file.Read(nullptr, 0)})
    {
        ASSERT_FALSE(read);
        EXPECT_EQ(read.error(), std::errc::bad_file_descriptor);
    }
    ASSERT_TRUE(file.Close());
    ExpectText(path, "old");
}

TEST(File, ReopenFailureState)
{
    using rad::FileAccess;
    using rad::FileOpenDisposition;
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    const auto path = directory.path / "reopen.bin";
    ASSERT_TRUE(rad::File::WriteText(path, "old"));
    rad::File file;
    ASSERT_TRUE(file.Open(path, FileAccess::Read));
    ASSERT_TRUE(file.Seek(1));

    // Invalid options must not close the file or move its cursor.
    const auto invalid = file.Open(path, FileAccess::Append, FileOpenDisposition::CreateAlways);
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.error(), std::errc::invalid_argument);
    char character{};
    const auto read = file.Read(&character, 1);
    ASSERT_TRUE(read) << read.error().message();
    ASSERT_EQ(read.value(), 1u);
    EXPECT_EQ(character, 'l');

    // Valid options start reopening; an OS error leaves the previous file closed.
    const auto missing = file.Open(directory.path / "missing.bin", FileAccess::Read);
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error(), std::errc::no_such_file_or_directory);
    EXPECT_FALSE(file.IsOpen());
}

TEST(File, OpenRejectsDirectories)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    rad::File file;
    const auto opened = file.Open(directory.path, rad::FileAccess::Read);
    ASSERT_FALSE(opened);
    EXPECT_FALSE(file.IsOpen());
#if !defined(RAD_OS_WINDOWS)
    EXPECT_EQ(opened.error(), std::errc::is_a_directory);
#endif
}

TEST(File, UpdateStreamSwitchesDirection)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    const auto path = directory.path / "update.bin";
    ASSERT_TRUE(rad::File::WriteText(path, "abcd"));

    rad::File file;
    ASSERT_TRUE(file.Open(path, rad::FileAccess::ReadWrite));
    char buffer[2]{};
    const auto initial = file.Read(buffer, sizeof(buffer));
    ASSERT_TRUE(initial);
    ASSERT_EQ(initial.value(), 2u);
    EXPECT_EQ(std::string_view(buffer, 2), "ab");

    // Overwrite at the logical read position, not at the end of the read-ahead buffer.
    ASSERT_TRUE(file.Write("X", 1));
    const auto tail = file.Read(buffer, sizeof(buffer));
    ASSERT_TRUE(tail);
    ASSERT_EQ(tail.value(), 1u);
    EXPECT_EQ(buffer[0], 'd');
    EXPECT_TRUE(file.Eof());
    const auto eof = file.Read(buffer, sizeof(buffer));
    ASSERT_TRUE(eof);
    EXPECT_EQ(eof.value(), 0u);

    rad::File moved = std::move(file);
    EXPECT_FALSE(file.IsOpen());
    ASSERT_TRUE(moved.Write("!", 1));
    EXPECT_FALSE(moved.Eof());
    const auto size = moved.Size();
    ASSERT_TRUE(size);
    EXPECT_EQ(size.value(), 5u);

    const auto end = moved.Read(buffer, 1);
    ASSERT_TRUE(end);
    EXPECT_EQ(end.value(), 0u);
    EXPECT_TRUE(moved.Eof());
    ASSERT_TRUE(moved.Seek(-1, rad::FileSeekOrigin::End));
    EXPECT_FALSE(moved.Eof());
    const auto last = moved.Read(buffer, 1);
    ASSERT_TRUE(last);
    ASSERT_EQ(last.value(), 1u);
    EXPECT_EQ(buffer[0], '!');

    constexpr std::int64_t largeOffset = 5LL * 1024 * 1024 * 1024;
    ASSERT_TRUE(moved.Seek(largeOffset));
    const auto position = moved.Position();
    ASSERT_TRUE(position);
    EXPECT_EQ(position.value(), static_cast<std::uint64_t>(largeOffset));
    ASSERT_TRUE(moved.Close());
    ExpectText(path, "abXd!");
}

TEST(File, BinaryTextAndLineEndings)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    const auto path = directory.path / rad::MakeFilePath("file-\xE4\xB8\xAD.txt");
    constexpr std::string_view text = "one\r\n\n\x1A\two";
    ASSERT_TRUE(rad::File::AppendText(path, text));
    ExpectText(path, text);

    rad::File file;
    ASSERT_TRUE(file.Open(path, rad::FileAccess::Read));
    const std::vector<std::string> expectedLines{"one", "", "\x1A\two"};
    std::string line = "stale";

    // Empty and unterminated lines are successful, and a previous error is cleared.
    {
        std::error_code ec = std::make_error_code(std::errc::io_error);
        for (const auto& expected : expectedLines)
        {
            ASSERT_TRUE(file.ReadLine(line, ec)) << ec.message();
            EXPECT_EQ(line, expected);
            EXPECT_FALSE(ec);
        }
        ec = std::make_error_code(std::errc::io_error);
        EXPECT_FALSE(file.ReadLine(line, ec));
        EXPECT_FALSE(ec);
        EXPECT_TRUE(line.empty());
    }

    // The throwing overload still returns false at normal EOF.
    EXPECT_FALSE(file.ReadLine(line));
    const auto lines = rad::File::ReadAllLines(path);
    ASSERT_TRUE(lines) << lines.error().message();
    EXPECT_EQ(lines.value(), expectedLines);
}

TEST(File, ReadLineReportsErrors)
{
    rad::File file;
    std::string line = "stale";
    std::error_code ec;
    EXPECT_FALSE(file.ReadLine(line, ec));
    EXPECT_EQ(ec, std::errc::bad_file_descriptor);
    EXPECT_TRUE(line.empty());

    EXPECT_THROW(static_cast<void>(file.ReadLine(line)), std::system_error);
}

#if defined(RAD_OS_LINUX)
TEST(File, BufferedWriteReportsNoSpace)
{
    if (!rad::Exists("/dev/full"))
    {
        GTEST_SKIP() << "/dev/full is unavailable";
    }
    rad::File file;

    // Close reports write errors even if it is the first operation to flush the buffer.
    {
        ASSERT_TRUE(file.Open("/dev/full", rad::FileAccess::Write));
        ASSERT_TRUE(file.Write("x", 1));
        const auto closed = file.Close();
        ASSERT_FALSE(closed);
        EXPECT_EQ(closed.error(), std::errc::no_space_on_device);
        EXPECT_FALSE(file.IsOpen());
    }

    // Seeking reports a write failure if it encounters one while flushing.
    {
        ASSERT_TRUE(file.Open("/dev/full", rad::FileAccess::ReadWrite));
        ASSERT_TRUE(file.Write("x", 1));
        const auto emptyRead = file.Read(nullptr, 0);
        ASSERT_TRUE(emptyRead);
        EXPECT_EQ(emptyRead.value(), 0u);

        const auto sought = file.Seek(0);
        ASSERT_FALSE(sought);
        EXPECT_EQ(sought.error(), std::errc::no_space_on_device);

        // A no-op is independent of the CRT's buffering behavior after a failed flush.
        EXPECT_TRUE(file.Write(nullptr, 0));
        static_cast<void>(file.Close());
        EXPECT_FALSE(file.IsOpen());
    }
}
#endif
