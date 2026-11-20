#include <rad/System/OS.h>

#include <rad/System/FileSystem.h>
#include <rad/System/TempDirectory.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

TEST(OS, ProcessAndPathQueries)
{
    EXPECT_GT(rad::os::getpid(), 0U);
    EXPECT_FALSE(rad::os::get_exec_path().empty());
    EXPECT_TRUE(rad::os::path::isfile(rad::os::executable_path()));
    EXPECT_FALSE(rad::ToUtf8(rad::os::executable_path()).empty());
    EXPECT_EQ(rad::os::executable_directory(), rad::os::executable_path().parent_path());
    EXPECT_TRUE(rad::os::path::isdir(rad::os::temp_directory_path()));

    EXPECT_TRUE(rad::os::path::isabs(rad::os::path::realpath("missing-component")));
    const auto [dotRoot, dotExtension] = rad::os::path::splitext("....jpg");
    EXPECT_EQ(dotRoot, "....jpg");
    EXPECT_TRUE(dotExtension.empty());

#if defined(RAD_OS_WINDOWS)
    EXPECT_EQ(rad::os::path::normcase(rad::os::path::commonpath({R"(C:\Foo\a)", R"(c:\foo\b)"})),
              R"(c:\foo)");
    EXPECT_THROW((void)rad::os::path::commonpath({R"(C:\foo)", R"(D:\foo)"}),
                 std::invalid_argument);

    const auto [drive, tail] = rad::os::path::splitdrive(R"(\\server\share\directory)");
    EXPECT_EQ(drive, R"(\\server\share)");
    EXPECT_EQ(tail, R"(\directory)");
    EXPECT_TRUE(rad::os::path::ismount(rad::os::getcwd().root_path()));
#endif
}

TEST(FileSystem, LastWriteTime)
{
    const auto directory = rad::TempDirectory::Create();
    ASSERT_TRUE(directory) << directory.error().message();

    // Both files and directories have modification times; missing paths do not.
    {
        const auto executable = rad::os::executable_path();
        const auto fileTime = rad::LastWriteTime(executable);
        ASSERT_TRUE(fileTime) << fileTime.error().message();
        EXPECT_EQ(fileTime.value(), std::filesystem::last_write_time(executable));

        const auto& path = directory.value().Path();
        const auto directoryTime = rad::LastWriteTime(path);
        ASSERT_TRUE(directoryTime) << directoryTime.error().message();
        EXPECT_EQ(directoryTime.value(), std::filesystem::last_write_time(path));

        const auto missing = rad::LastWriteTime(path / "missing");
        ASSERT_FALSE(missing);
        EXPECT_EQ(missing.error(), std::errc::no_such_file_or_directory);
    }

    // A native API must not query an existing path by truncating at an embedded NUL.
    {
        auto invalid = directory.value().Path().native();
        invalid.push_back(rad::FilePath::value_type{});
        invalid += rad::FilePath("suffix").native();
        const auto invalidTime = rad::LastWriteTime(invalid);
        ASSERT_FALSE(invalidTime);
        EXPECT_EQ(invalidTime.error(), std::errc::invalid_argument);
    }
}

TEST(FileSystem, QueriesFileSizeAndType)
{
    const auto directory = rad::TempDirectory::Create();
    ASSERT_TRUE(directory) << directory.error().message();
    const auto& root = directory.value().Path();
    const auto file = root / "file";
    ASSERT_TRUE(std::ofstream(file, std::ios::binary) << "abc");

    // Existence includes directories; regular-file queries do not.
    {
        EXPECT_TRUE(rad::Exists(root));
        EXPECT_FALSE(rad::IsRegularFile(root));
        EXPECT_TRUE(rad::Exists(file));
        EXPECT_TRUE(rad::IsRegularFile(file));
        const auto size = rad::GetSize(file);
        ASSERT_TRUE(size) << size.error().message();
        EXPECT_EQ(size.value(), 3U);
    }

    // Missing paths and type mismatches are negative queries, not errors.
    {
        std::error_code ec;
        auto invalid = file.native();
        invalid.push_back(rad::FilePath::value_type{});
        EXPECT_FALSE(rad::Exists(invalid, ec));
        EXPECT_EQ(ec, std::errc::invalid_argument);
        const auto missing = root / "missing";
        EXPECT_FALSE(rad::Exists(missing, ec));
        EXPECT_FALSE(ec);

        EXPECT_FALSE(rad::IsRegularFile(invalid, ec));
        EXPECT_EQ(ec, std::errc::invalid_argument);
        EXPECT_FALSE(rad::IsRegularFile(root, ec));
        EXPECT_FALSE(ec);

        const auto invalidSize = rad::GetSize(invalid);
        ASSERT_FALSE(invalidSize);
        EXPECT_EQ(invalidSize.error(), std::errc::invalid_argument);
        const auto missingSize = rad::GetSize(missing);
        ASSERT_FALSE(missingSize);
        EXPECT_EQ(missingSize.error(), std::errc::no_such_file_or_directory);
    }
}

TEST(FileSystem, CopiesRenamesAndRemovesEntries)
{
    const auto directory = rad::TempDirectory::Create();
    ASSERT_TRUE(directory) << directory.error().message();
    const auto& root = directory.value().Path();
    const auto source = root / "source";
    const auto copy = root / "copy";
    const auto renamed = root / "renamed";

    // A copy must not overwrite unless requested.
    {
        ASSERT_TRUE(std::ofstream(source, std::ios::binary) << "abc");
        ASSERT_TRUE(rad::Copy(source, copy));
        ASSERT_TRUE(std::ofstream(source, std::ios::binary) << "longer");
        const auto conflict = rad::Copy(source, copy);
        ASSERT_FALSE(conflict);
        EXPECT_EQ(conflict.error(), std::errc::file_exists);
        EXPECT_EQ(std::filesystem::file_size(copy), 3U);
        ASSERT_TRUE(rad::Copy(source, copy, true));
        EXPECT_EQ(std::filesystem::file_size(copy), 6U);
    }

    // Rename moves the entry; removal is non-recursive and reports absence.
    {
        ASSERT_TRUE(rad::Rename(copy, renamed));
        EXPECT_FALSE(std::filesystem::exists(copy));
        EXPECT_TRUE(std::filesystem::is_regular_file(renamed));
        const auto invalidRename = rad::Rename(renamed, {});
        ASSERT_FALSE(invalidRename);
        EXPECT_EQ(invalidRename.error(), std::errc::invalid_argument);
        const auto nonEmpty = rad::Remove(root);
        ASSERT_FALSE(nonEmpty);
        EXPECT_EQ(nonEmpty.error(), std::errc::directory_not_empty);
        EXPECT_TRUE(rad::Remove(renamed));
        const auto missing = rad::Remove(renamed);
        ASSERT_FALSE(missing);
        EXPECT_EQ(missing.error(), std::errc::no_such_file_or_directory);

        const auto empty = root / "empty";
        ASSERT_TRUE(std::filesystem::create_directory(empty));
        EXPECT_TRUE(rad::Remove(empty));
    }
}
