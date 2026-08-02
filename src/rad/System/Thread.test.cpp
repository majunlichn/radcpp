#include <rad/System/Thread.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <thread>

TEST(System, Thread)
{
    constexpr std::string_view threadName = "worker";

    std::uint64_t workerThreadId = 0;
    std::string threadNameRetrieved;
    bool nameWasSet = false;
    std::thread worker(
        [&]
        {
            workerThreadId = rad::GetCurrentThreadId();
            nameWasSet = rad::SetThreadName(threadName);
            threadNameRetrieved = rad::GetThreadName();
        });
    worker.join();

    EXPECT_NE(rad::GetCurrentThreadId(), 0);
    EXPECT_NE(workerThreadId, 0);
    EXPECT_NE(workerThreadId, rad::GetCurrentThreadId());
    EXPECT_TRUE(nameWasSet);
    EXPECT_EQ(threadNameRetrieved, threadName);
}
