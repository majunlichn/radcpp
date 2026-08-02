#include <rad/System/Process.h>

#include <gtest/gtest.h>

TEST(System, Process)
{
    const std::string output =
        rad::Process::ExecuteAndCaptureOutput("cmake", {"-E", "echo", "Hello, World!"});

    EXPECT_EQ(output, "Hello, World!\n");
}
