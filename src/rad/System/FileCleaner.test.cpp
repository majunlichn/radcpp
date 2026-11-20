#include <rad/System/FileCleaner.h>

#include <rad/Core/Platform.h>
#include <rad/System/TempDirectory.h>

#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <thread>
#include <vector>

TEST(FileCleaner, RemovesOnlyRegisteredFiles)
{
    const auto directory = rad::TempDirectory::Create(".");
    ASSERT_TRUE(directory) << directory.error().message();
    const auto root = std::filesystem::relative(directory.value().Path());
    const auto remove = root / "remove.txt";
    const auto keep = root / "keep.txt";
    ASSERT_TRUE(std::ofstream(remove));
    ASSERT_TRUE(std::ofstream(keep));
    const auto link = root / "keep-link.txt";
    std::filesystem::create_hard_link(keep, link);

    {
        rad::FileCleaner cleaner;
        cleaner.Register(remove);
        cleaner.Register(link);
        const auto absoluteKeep = std::filesystem::absolute(keep);
        EXPECT_FALSE(cleaner.IsRegistered(keep));
        cleaner.Register(keep);
        EXPECT_TRUE(cleaner.IsRegistered(absoluteKeep));
        cleaner.Register(absoluteKeep);
        cleaner.Unregister(absoluteKeep);
        EXPECT_FALSE(cleaner.IsRegistered(keep));
        cleaner.Unregister(keep);
        EXPECT_TRUE(std::filesystem::exists(remove));
    }
    EXPECT_FALSE(std::filesystem::exists(remove));
    EXPECT_FALSE(std::filesystem::exists(link));
    EXPECT_TRUE(std::filesystem::exists(keep));
}

#if defined(RAD_OS_WINDOWS)
TEST(FileCleaner, CleanRetriesLockedFileAndRemovesOtherFiles)
{
    const auto directory = rad::TempDirectory::Create(".");
    ASSERT_TRUE(directory) << directory.error().message();
    const auto& root = directory.value().Path();
    const auto lockedPath = root / "a-locked";
    const auto file = root / "b-file";
    const auto missing = root / "c-missing";
    ASSERT_TRUE(std::ofstream(lockedPath));
    ASSERT_TRUE(std::ofstream(file));

    rad::FileCleaner cleaner;
    std::ifstream locked(lockedPath);
    ASSERT_TRUE(locked);
    cleaner.Register(lockedPath);
    cleaner.Register(file);
    cleaner.Register(missing);
    EXPECT_FALSE(cleaner.Clean());
    EXPECT_TRUE(std::filesystem::exists(lockedPath));
    EXPECT_FALSE(std::filesystem::exists(file));
    EXPECT_TRUE(cleaner.IsRegistered(lockedPath));
    EXPECT_FALSE(cleaner.IsRegistered(file));
    EXPECT_FALSE(cleaner.IsRegistered(missing));
    locked.close();
    EXPECT_TRUE(cleaner.Clean());
    EXPECT_FALSE(std::filesystem::exists(lockedPath));
}
#endif

TEST(FileCleaner, LeavesDirectoriesIntact)
{
    const auto directory = rad::TempDirectory::Create(".");
    ASSERT_TRUE(directory) << directory.error().message();
    const auto& root = directory.value().Path();
    const auto tree = root / "tree";
    const auto child = tree / "child.txt";
    const auto empty = root / "empty";
    const auto missing = root / "missing";
    ASSERT_TRUE(std::ofstream(tree));
    ASSERT_TRUE(std::filesystem::create_directory(empty));

    rad::FileCleaner cleaner;
    cleaner.Register(tree);
    cleaner.Register(empty);
    cleaner.Register(missing);
    EXPECT_TRUE(cleaner.IsRegistered(missing));
    // A file path can be replaced by a directory between registration and cleanup.
    ASSERT_TRUE(std::filesystem::remove(tree));
    ASSERT_TRUE(std::filesystem::create_directory(tree));
    ASSERT_TRUE(std::ofstream(child));
    EXPECT_FALSE(cleaner.Clean());
    EXPECT_TRUE(std::filesystem::is_regular_file(child));
    EXPECT_TRUE(std::filesystem::is_directory(empty));
    EXPECT_FALSE(cleaner.IsRegistered(missing));
    EXPECT_TRUE(cleaner.IsRegistered(tree));
    EXPECT_TRUE(cleaner.IsRegistered(empty));
    cleaner.Unregister(tree);
    cleaner.Unregister(empty);
}

#if !defined(RAD_OS_WINDOWS)
TEST(FileCleaner, UnlinksSymlinksWithoutFollowingThem)
{
    const auto directory = rad::TempDirectory::Create(".");
    ASSERT_TRUE(directory) << directory.error().message();
    const auto& root = directory.value().Path();
    const auto target = root / "target";
    const auto link = root / "link";
    const auto dangling = root / "dangling";
    ASSERT_TRUE(std::filesystem::create_directory(target));
    ASSERT_TRUE(std::ofstream(target / "keep.txt"));
    std::filesystem::create_directory_symlink(target, link);
    std::filesystem::create_symlink("missing.txt", dangling);

    rad::FileCleaner cleaner;
    cleaner.Register(link);
    cleaner.Register(dangling);
    EXPECT_TRUE(cleaner.Clean());
    EXPECT_FALSE(std::filesystem::is_symlink(std::filesystem::symlink_status(link)));
    EXPECT_FALSE(std::filesystem::is_symlink(std::filesystem::symlink_status(dangling)));
    EXPECT_TRUE(std::filesystem::is_regular_file(target / "keep.txt"));
}
#endif

TEST(FileCleaner, ConcurrentRegisterAndUnregister)
{
    const auto directory = rad::TempDirectory::Create(".");
    ASSERT_TRUE(directory) << directory.error().message();
    const auto& root = directory.value().Path();
    const auto remove = root / "remove.txt";
    const auto keep = root / "keep.txt";
    ASSERT_TRUE(std::ofstream(remove));
    ASSERT_TRUE(std::ofstream(keep));

    {
        rad::FileCleaner cleaner;
        cleaner.Register(keep);
        {
            std::vector<std::jthread> workers;
            for (int i = 0; i < 8; ++i)
            {
                workers.emplace_back(
                    [&]
                    {
                        cleaner.Register(remove);
                        cleaner.Unregister(keep);
                    });
            }
        }
        EXPECT_TRUE(cleaner.IsRegistered(remove));
        EXPECT_FALSE(cleaner.IsRegistered(keep));
    }
    EXPECT_FALSE(std::filesystem::exists(remove));
    EXPECT_TRUE(std::filesystem::exists(keep));
}

TEST(FileCleaner, RejectsRootsAndDotEntries)
{
    const auto checkRejected = [](const rad::FilePath& path)
    {
        auto cleaner = std::make_unique<rad::FileCleaner>();
        SCOPED_TRACE(path.string());
        try
        {
            cleaner->Register(path);
            // Never run cleanup on an unexpectedly accepted root or working directory.
            static_cast<void>(cleaner.release());
            ADD_FAILURE() << "Expected invalid_argument";
        }
        catch (const std::system_error& error)
        {
            EXPECT_EQ(error.code(), std::errc::invalid_argument);
        }
    };

    checkRejected(".");
    checkRejected(rad::FilePath("..") / "");
    checkRejected(std::filesystem::current_path().root_path());
#if defined(RAD_OS_WINDOWS)
    checkRejected(LR"(\\server\share)");
    checkRejected(LR"(\\?\UNC\server\share\)");
    checkRejected(LR"(\\?\C:\)");
    checkRejected(LR"(\\?\Volume{00000000-0000-0000-0000-000000000000}\)");
#endif
}
