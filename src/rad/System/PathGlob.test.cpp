#include <rad/System/FileSystem.h>

#include <rad/Core/Platform.h>
#include <rad/Core/String.h>
#include <rad/IO/File.h>
#include <rad/System/OS.h>

#include <gtest/gtest.h>

#include <initializer_list>

#if defined(RAD_OS_WINDOWS)
#include <windows.h>
#endif

namespace
{

struct TestDirectory
{
    rad::FilePath path = "rad-glob-test-" + rad::ToHexString(rad::os::urandom(16));
    bool created = std::filesystem::create_directory(path);

    ~TestDirectory()
    {
        if (created)
        {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    }

    void AddFile(std::string_view relative) const
    {
        const auto file = path / rad::MakeFilePath(relative);
        std::filesystem::create_directories(file.parent_path());
        ASSERT_TRUE(rad::File::WriteText(file, {})) << file;
    }
};

void ExpectGlob(const rad::FilePath& root, std::string_view pattern,
                std::initializer_list<std::string_view> expected,
                const rad::PathGlobOptions& options = {})
{
    SCOPED_TRACE(pattern);
    const auto result = rad::PathGlob(root, pattern, options);
    ASSERT_TRUE(result) << result.error().message();
    std::vector<std::string> paths;
    for (const auto& path : result.value())
    {
        const auto utf8 = path.generic_u8string();
        paths.emplace_back(utf8.begin(), utf8.end());
    }
    EXPECT_EQ(paths, std::vector<std::string>(expected.begin(), expected.end()));
}

void ExpectGlobError(const rad::FilePath& root, std::string_view pattern, std::errc expected)
{
    SCOPED_TRACE(::testing::Message() << root << ", pattern: " << pattern);
    const auto result = rad::PathGlob(root, pattern);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), expected);
}

} // namespace

TEST(PathGlob, MatchesShellPatterns)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    for (const auto* path :
         {"data/aX.txt", "data/b2.txt", "data/a2.txt", "data/a1.txt", "data/a4.TXT", "data/[1].txt",
          "data/].txt", "data/-.txt", "data/\U0001f600.md", "data/\u732b.md", "data/ab.md",
          "data/nested/a3.txt", "data/plain.txt", "data/plain-old.txt"})
    {
        ASSERT_NO_FATAL_FAILURE(directory.AddFile(path));
    }

    // Segment boundaries, case sensitivity, and deterministic ordering.
    {
        ExpectGlob(directory.path, "./data//a[0-9].txt", {"data/a1.txt", "data/a2.txt"});
        ExpectGlob(directory.path, "DATA/*.txt", {});
        ExpectGlob(directory.path, "data/[ab][!X].txt",
                   {"data/a1.txt", "data/a2.txt", "data/b2.txt"});
        ExpectGlob(directory.path, "data/a**.txt", {"data/a1.txt", "data/a2.txt", "data/aX.txt"});
        ExpectGlob(directory.path, "data/plain*.txt", {"data/plain-old.txt", "data/plain.txt"});
    }

    // Escapes, literal brackets and hyphens, and Unicode code-point matching.
    {
        ExpectGlob(directory.path, R"(data/\[1\].txt)", {"data/[1].txt"});
        ExpectGlob(directory.path, "data/[]-].txt", {"data/-.txt", "data/].txt"});
        ExpectGlob(directory.path, "data/?.md", {"data/\u732b.md", "data/\U0001f600.md"});
        ExpectGlob(directory.path, "data/[\u732b].md", {"data/\u732b.md"});
    }
}

TEST(PathGlob, TraversesGlobstarsAndDotDirectories)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    for (const auto* path : {"hit.txt", "src/hit.txt", "src/src/deep/hit.txt", ".cache/hit.txt",
                             "src/.hidden/hit.txt", "src/.hit.txt"})
    {
        ASSERT_NO_FATAL_FAILURE(directory.AddFile(path));
    }
    ASSERT_TRUE(std::filesystem::create_directory(directory.path / "empty"));

    // Globstars match zero levels, merge overlapping matches, and do not return the root.
    {
        ExpectGlob(directory.path, "**/*hit.txt",
                   {"hit.txt", "src/hit.txt", "src/src/deep/hit.txt"});
        ExpectGlob(directory.path, "**/src/**/hit.txt", {"src/hit.txt", "src/src/deep/hit.txt"});
        ExpectGlob(directory.path, "**/", {"empty", "src", "src/src", "src/src/deep"});
        ExpectGlob(directory.path, "empty/**", {"empty"});
        ExpectGlob(directory.path, "hit.txt/**", {});
    }

    // Leading dot behavior applies per segment, even when a globstar is active.
    {
        ExpectGlob(directory.path, "**/.*/hit.txt", {".cache/hit.txt", "src/.hidden/hit.txt"});
        ExpectGlob(directory.path, "src/[.]hit.txt", {});
        ExpectGlob(directory.path, "src/.*.txt", {"src/.hit.txt"});
        ExpectGlob(directory.path, "**/*hit.txt",
                   {".cache/hit.txt", "hit.txt", "src/.hidden/hit.txt", "src/.hit.txt",
                    "src/hit.txt", "src/src/deep/hit.txt"},
                   {.includeDotFiles = true});
    }
}

TEST(PathGlob, DoesNotTraverseDirectoryLinks)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    ASSERT_NO_FATAL_FAILURE(directory.AddFile("target/file.txt"));
    std::error_code error;
    std::filesystem::create_directory_symlink(std::filesystem::absolute(directory.path / "target"),
                                              directory.path / "alias", error);
    if (error == std::errc::permission_denied || error == std::errc::operation_not_permitted ||
        error == std::errc::operation_not_supported
#if defined(RAD_OS_WINDOWS)
        || error == std::error_code(ERROR_PRIVILEGE_NOT_HELD, std::system_category())
#endif
    )
    {
        GTEST_SKIP() << "Directory symlinks are unavailable: " << error.message();
    }
    ASSERT_FALSE(error) << error.message();
    ExpectGlob(directory.path, "**", {"alias", "target", "target/file.txt"});
    ExpectGlob(directory.path, "alias/**/file.txt", {});
    ExpectGlob(directory.path, "alias/", {});
}

TEST(PathGlob, ReportsInvalidPatternsAndRoots)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    ASSERT_NO_FATAL_FAILURE(directory.AddFile("file.txt"));

    // Syntax errors and invalid encodings are distinct.
    {
        for (const auto pattern : std::initializer_list<std::string_view>{"",
                                                                          {"*\0*", 3},
                                                                          "/x",
                                                                          "./D:/x",
                                                                          "a/../x",
                                                                          "a/./x",
                                                                          "abc\\",
                                                                          "[a",
                                                                          "[z-a]",
                                                                          "[[:alpha:]]"})
        {
            ExpectGlobError(directory.path, pattern, std::errc::invalid_argument);
        }
        ExpectGlobError(directory.path, "\xc0\xaf", std::errc::illegal_byte_sequence);
    }

    // A bad root is an error, unlike a valid pattern with no match.
    {
        ExpectGlob(directory.path, "missing*", {});
        ExpectGlobError({}, "*", std::errc::invalid_argument);
        ExpectGlobError(directory.path / "missing", "*", std::errc::no_such_file_or_directory);
        ExpectGlobError(directory.path / "file.txt", "*", std::errc::not_a_directory);
    }
}

#if defined(RAD_OS_LINUX)
TEST(PathGlob, RejectsInvalidUtf8OnlyInScannedDirectories)
{
    const TestDirectory directory;
    ASSERT_TRUE(directory.created);
    ASSERT_NO_FATAL_FAILURE(directory.AddFile("other/\xff"));
    ASSERT_NO_FATAL_FAILURE(directory.AddFile("data/file.txt"));

    ExpectGlob(directory.path, "data/*.txt", {"data/file.txt"});
    ExpectGlobError(directory.path, "**", std::errc::illegal_byte_sequence);
}
#endif
