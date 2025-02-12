#include <radcpp/System/Time.h>
#include <radcpp/IO/Logging.h>
#include <cmath>
#include <thread>
#include <gtest/gtest.h>

TEST(System, Time)
{
    RAD_LOG_DEFAULT(info, "Time UTC: {}", rad::GetTimeStringUTC());
    RAD_LOG_DEFAULT(info, "Time ISO8601: {}", rad::GetTimeStringISO8601());

    rad::Stopwatch watch;
    watch.Start();
    std::this_thread::sleep_for(rad::Milliseconds(100));
    watch.Stop();
    double epsilon = std::abs(watch.GetElapsedSeconds() - 0.1);
    EXPECT_TRUE(epsilon < 0.01);
    epsilon = std::abs(watch.GetElapsedMilliseconds() - 100);
    EXPECT_TRUE(epsilon < 10);
}
