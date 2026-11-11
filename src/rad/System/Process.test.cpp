#include <rad/System/Process.h>

#include <gtest/gtest.h>

TEST(Process, ExecuteAndCaptureOutput)
{
    const std::string output =
        rad::Process::ExecuteAndCaptureOutput("cmake", {"-E", "echo", "Hello, World!"});

    EXPECT_EQ(output, "Hello, World!\n");
}
