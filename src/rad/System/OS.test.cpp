#include <rad/System/OS.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

TEST(System, OS)
{
    EXPECT_GT(rad::os::getpid(), 0U);
    EXPECT_FALSE(rad::os::get_exec_path().empty());
    EXPECT_TRUE(rad::os::path::isfile(rad::os::executable_path()));
    EXPECT_FALSE(rad::PathToUtf8(rad::os::executable_path()).empty());
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
