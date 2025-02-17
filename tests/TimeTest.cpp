#include <radcpp/System/Time.h>
#include <radcpp/IO/Logging.h>
#include <cmath>
#include <thread>
#include <gtest/gtest.h>

TEST(System, Time)
{
    LOG_DEFAULT(info, "Time UTC: {}", rad::GetTimeStringUTC());
    LOG_DEFAULT(info, "Time ISO8601: {}", rad::GetTimeStringISO8601());

    rad::Stopwatch watch;
    watch.Start();
    std::this_thread::sleep_for(rad::Milliseconds(100));
    watch.Stop();
    double elapsedSeconds = watch.GetElapsedSeconds();
    EXPECT_TRUE(elapsedSeconds > 0.1);
    EXPECT_TRUE(elapsedSeconds < 0.15);
    double elapsedMilliseconds = watch.GetElapsedMilliseconds();
    EXPECT_TRUE(elapsedMilliseconds > 100);
    EXPECT_TRUE(elapsedMilliseconds < 150);
}
