#include <rad/System/Time.h>

#include <gtest/gtest.h>

#include <iostream>
#include <limits>
#include <regex>
#include <ratio>
#include <thread>
#include <type_traits>

TEST(System, Time)
{
    const std::string utc = rad::GetTimeStringUTC();
    const std::string local = rad::GetLocalTimeStringISO8601();

    std::cout << "UTC: " << utc << '\n';
    std::cout << "Local ISO 8601: " << local << '\n';

    const std::regex utcPattern{R"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)"};
    const std::regex localPattern{R"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2})"};

    EXPECT_TRUE(std::regex_match(utc, utcPattern));
    EXPECT_TRUE(std::regex_match(local, localPattern));
}

TEST(System, TimeFormatting)
{
    const auto time = std::chrono::system_clock::time_point{rad::Milliseconds{123}};

    EXPECT_EQ(rad::GetTimeStringUTC(time), "1970-01-01T00:00:00.123Z");

    const std::string local = rad::GetLocalTimeStringISO8601(time);
    const auto parsedLocal = rad::ParseTimeISO8601(local);
    ASSERT_TRUE(parsedLocal.has_value());
    EXPECT_EQ(std::chrono::time_point_cast<rad::Milliseconds>(*parsedLocal), time);
}

TEST(System, ParseTimeISO8601)
{
    const auto epoch = std::chrono::system_clock::time_point{};
    EXPECT_EQ(rad::ParseTimeISO8601("1970-01-01T00:00:00Z"), epoch);
    EXPECT_EQ(rad::ParseTimeISO8601("1970-01-01T08:00:00.123+08:00"),
              epoch + rad::Milliseconds{123});
    EXPECT_EQ(rad::ParseTimeISO8601("1969-12-31T19:00:00-05:00"), epoch);

    EXPECT_FALSE(rad::ParseTimeISO8601("2026-07-23T10:00:00"));
    EXPECT_FALSE(rad::ParseTimeISO8601("2026-02-29T10:00:00Z"));
    EXPECT_FALSE(rad::ParseTimeISO8601("2026-07-23T10:00:00.1234567890Z"));
    EXPECT_FALSE(rad::ParseTimeISO8601("2026-07-23T10:00:00+14:01"));
    EXPECT_FALSE(rad::ParseTimeISO8601("2026-07-23T10:00:00." + std::string(100, '1') + "Z"));
}

TEST(System, UnixTimeMilliseconds)
{
    const auto time = rad::FromUnixTimeMilliseconds(1234);
    ASSERT_TRUE(time.has_value());
    EXPECT_EQ(std::chrono::duration_cast<rad::Milliseconds>(time->time_since_epoch()).count(),
              1234);

    using ClockDuration = std::chrono::system_clock::duration;
    if constexpr (std::ratio_less_equal_v<ClockDuration::period, rad::Milliseconds::period>)
    {
        const auto minimum = std::chrono::ceil<rad::Milliseconds>(ClockDuration::min()).count();
        const auto maximum = std::chrono::floor<rad::Milliseconds>(ClockDuration::max()).count();
        EXPECT_TRUE(rad::FromUnixTimeMilliseconds(minimum));
        EXPECT_TRUE(rad::FromUnixTimeMilliseconds(maximum));

        if constexpr (std::ratio_less_v<ClockDuration::period, rad::Milliseconds::period>)
        {
            EXPECT_FALSE(rad::FromUnixTimeMilliseconds(std::numeric_limits<std::int64_t>::min()));
            EXPECT_FALSE(rad::FromUnixTimeMilliseconds(std::numeric_limits<std::int64_t>::max()));
        }
    }

    const auto before = std::chrono::floor<rad::Milliseconds>(std::chrono::system_clock::now())
                            .time_since_epoch()
                            .count();
    const std::int64_t current = rad::GetUnixTimeMilliseconds();
    const auto after = std::chrono::floor<rad::Milliseconds>(std::chrono::system_clock::now())
                           .time_since_epoch()
                           .count();
    EXPECT_GE(current, before);
    EXPECT_LE(current, after);
}

TEST(System, Stopwatch)
{
    rad::Stopwatch stopwatch;
    static_assert(std::is_same_v<decltype(stopwatch.Elapsed()), rad::Seconds>);

    EXPECT_FALSE(stopwatch.IsRunning());
    EXPECT_EQ(stopwatch.ElapsedMilliseconds(), rad::Milliseconds::zero());

    stopwatch.Start();
    EXPECT_TRUE(stopwatch.IsRunning());
    std::this_thread::sleep_for(rad::Milliseconds{2});
    stopwatch.Stop();

    EXPECT_FALSE(stopwatch.IsRunning());
    const rad::Milliseconds firstElapsed = stopwatch.ElapsedMilliseconds();
    EXPECT_GE(firstElapsed, rad::Milliseconds{1});
    EXPECT_EQ(stopwatch.Elapsed<rad::Milliseconds>(), firstElapsed);

    std::this_thread::sleep_for(rad::Milliseconds{2});
    EXPECT_EQ(stopwatch.ElapsedMilliseconds(), firstElapsed);

    stopwatch.Start();
    std::this_thread::sleep_for(rad::Milliseconds{2});
    stopwatch.Stop();
    EXPECT_GT(stopwatch.ElapsedMilliseconds(), firstElapsed);
}

TEST(System, Deadline)
{
    const rad::Deadline active{rad::Seconds{1}};
    EXPECT_FALSE(active.IsExpired());
    EXPECT_GT(active.Remaining(), rad::PerfClock::duration::zero());
    EXPECT_LE(active.Remaining(), rad::Seconds{1});

    const rad::Deadline expired{rad::PerfClock::duration::zero()};
    EXPECT_TRUE(expired.IsExpired());
    EXPECT_EQ(expired.Remaining(), rad::PerfClock::duration::zero());

    const rad::Deadline negative{rad::Seconds{-1}};
    EXPECT_TRUE(negative.IsExpired());
    EXPECT_EQ(negative.Remaining(), rad::PerfClock::duration::zero());
}
