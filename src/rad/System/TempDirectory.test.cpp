#include <rad/System/TempDirectory.h>

#include <rad/Core/Platform.h>
#include <rad/System/OS.h>

#include <gtest/gtest.h>

#include <fstream>
#include <utility>

TEST(TempDirectory, OwnsTreeUntilCleanupOrRelease)
{
    const auto directory = rad::TempDirectory::Create();
    ASSERT_TRUE(directory) << directory.error().message();
    const auto& parent = directory.value().Path();
    EXPECT_TRUE(std::filesystem::equivalent(parent.parent_path(), rad::os::temp_directory_path()));

    // Move assignment removes the previous tree and transfers the new one.
    {
        rad::FilePath path;
        auto source = rad::TempDirectory::Create(parent);
        ASSERT_TRUE(source) << source.error().message();
        {
            auto destination = rad::TempDirectory::Create(parent);
            ASSERT_TRUE(destination) << destination.error().message();
            path = source.value().Path();
            const auto oldPath = destination.value().Path();
            ASSERT_NE(path, oldPath);
            EXPECT_EQ(path.parent_path(), parent);
            ASSERT_TRUE(std::filesystem::create_directory(path / "nested"));
            ASSERT_TRUE(std::ofstream(path / "nested" / "data.bin"));
            destination.value() = std::move(source.value());
            EXPECT_FALSE(std::filesystem::exists(oldPath));
            ASSERT_TRUE(source.value().Clean());
            EXPECT_TRUE(std::filesystem::exists(path / "nested" / "data.bin"));
        }
        EXPECT_FALSE(std::filesystem::exists(path));
    }

#if !defined(RAD_OS_WINDOWS)
    // Recursive cleanup removes the link, not its directory target.
    {
        const auto outside = parent / "outside";
        ASSERT_TRUE(std::filesystem::create_directory(outside));
        ASSERT_TRUE(std::ofstream(outside / "keep"));
        auto created = rad::TempDirectory::Create(parent);
        ASSERT_TRUE(created) << created.error().message();
        std::filesystem::create_directory_symlink(outside, created.value().Path() / "link");
        ASSERT_TRUE(created.value().Clean());
        EXPECT_TRUE(std::filesystem::is_regular_file(outside / "keep"));
    }
#endif

    // Release preserves the tree; explicit cleanup of a missing tree is harmless.
    {
        auto created = rad::TempDirectory::Create(parent);
        ASSERT_TRUE(created) << created.error().message();
        auto& directoryOwner = created.value();
        const auto path = directoryOwner.Release();
        ASSERT_TRUE(directoryOwner.Clean());
        EXPECT_TRUE(std::filesystem::is_directory(path));

        auto missing = rad::TempDirectory::Create(parent);
        ASSERT_TRUE(missing) << missing.error().message();
        ASSERT_TRUE(std::filesystem::remove(missing.value().Path()));
        ASSERT_TRUE(missing.value().Clean());
    }
}

TEST(TempDirectory, RejectsInvalidParents)
{
    const auto directory = rad::TempDirectory::Create();
    ASSERT_TRUE(directory) << directory.error().message();
    const auto& parent = directory.value().Path();

    // No directory is created when the parent is missing or would be truncated by the OS.
    {
        const auto missing = parent / "missing";
        const auto subdirectory = rad::TempDirectory::Create(missing);
        ASSERT_FALSE(subdirectory);
        EXPECT_EQ(subdirectory.error(), std::errc::no_such_file_or_directory);
        EXPECT_FALSE(std::filesystem::exists(missing));

        auto invalid = parent.native();
        invalid.push_back(rad::FilePath::value_type{});
        const auto invalidDirectory = rad::TempDirectory::Create(invalid);
        ASSERT_FALSE(invalidDirectory);
        EXPECT_EQ(invalidDirectory.error(), std::errc::invalid_argument);
    }
}
