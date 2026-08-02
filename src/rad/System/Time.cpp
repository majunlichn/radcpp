#include <rad/System/Time.h>

#include <chrono>
#include <cstdio>
#include <ratio>
#include <string_view>

namespace rad
{
namespace
{

struct tm* UniversalTime(const time_t* timer, struct tm* buffer)
{
    if (timer == nullptr || buffer == nullptr)
    {
        return nullptr;
    }

#if defined(_WIN32)
    return gmtime_s(buffer, timer) == 0 ? buffer : nullptr;
#else
    return gmtime_r(timer, buffer);
#endif
}

std::string FormatTime(const struct tm& time, long long milliseconds, std::string_view suffix)
{
    char buffer[32];
    const int length =
        std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02dT%02d:%02d:%02d.%03lld%.*s",
                      time.tm_year + 1900, time.tm_mon + 1, time.tm_mday, time.tm_hour, time.tm_min,
                      time.tm_sec, milliseconds, static_cast<int>(suffix.size()), suffix.data());
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(buffer))
    {
        return {};
    }

    return std::string(buffer, static_cast<std::size_t>(length));
}

struct TimeParts
{
    time_t timer;
    long long milliseconds;
};

TimeParts GetTimeParts(std::chrono::system_clock::time_point time)
{
    const auto seconds = std::chrono::floor<std::chrono::seconds>(time);
    return {
        std::chrono::system_clock::to_time_t(seconds),
        std::chrono::duration_cast<std::chrono::milliseconds>(time - seconds).count(),
    };
}

bool GetWallTime(const struct tm& time, std::chrono::sys_seconds& wallTime)
{
    const std::chrono::year_month_day date{
        std::chrono::year{time.tm_year + 1900},
        std::chrono::month{static_cast<unsigned int>(time.tm_mon + 1)},
        std::chrono::day{static_cast<unsigned int>(time.tm_mday)},
    };
    if (!date.ok())
    {
        return false;
    }

    wallTime = std::chrono::sys_days{date} + std::chrono::hours{time.tm_hour} +
               std::chrono::minutes{time.tm_min} + std::chrono::seconds{time.tm_sec};
    return true;
}

std::string GetUtcOffset(const struct tm& localTime, const struct tm& utcTime)
{
    std::chrono::sys_seconds localWallTime;
    std::chrono::sys_seconds utcWallTime;
    if (!GetWallTime(localTime, localWallTime) || !GetWallTime(utcTime, utcWallTime))
    {
        return {};
    }

    const auto offsetSeconds =
        std::chrono::duration_cast<std::chrono::seconds>(localWallTime - utcWallTime).count();
    if (offsetSeconds % 60 != 0)
    {
        return {};
    }

    const auto offsetMinutes = offsetSeconds / 60;
    const auto absoluteMinutes = offsetMinutes < 0 ? -offsetMinutes : offsetMinutes;
    char offset[7];
    const int length = std::snprintf(
        offset, sizeof(offset), "%c%02lld:%02lld", offsetMinutes < 0 ? '-' : '+',
        static_cast<long long>(absoluteMinutes / 60), static_cast<long long>(absoluteMinutes % 60));
    return length == 6 ? std::string(offset, 6) : std::string{};
}

bool ParseNumber(std::string_view value, std::size_t pos, std::size_t length, int& result)
{
    if (pos + length > value.size())
    {
        return false;
    }

    result = 0;
    for (std::size_t i = pos; i < pos + length; ++i)
    {
        if (value[i] < '0' || value[i] > '9')
        {
            return false;
        }
        result = result * 10 + value[i] - '0';
    }
    return true;
}

} // namespace

void Stopwatch::Start() noexcept
{
    if (!m_running)
    {
        m_start = PerfClock::now();
        m_running = true;
    }
}

void Stopwatch::Stop() noexcept
{
    if (m_running)
    {
        m_elapsed += PerfClock::now() - m_start;
        m_running = false;
    }
}

bool Stopwatch::IsRunning() const noexcept
{
    return m_running;
}

Milliseconds Stopwatch::ElapsedMilliseconds() const
{
    return Elapsed<Milliseconds>();
}

Deadline::Deadline(PerfClock::duration timeout) noexcept
{
    const PerfClock::time_point now = PerfClock::now();
    if (timeout <= PerfClock::duration::zero())
    {
        m_deadline = now;
        return;
    }

    const PerfClock::duration maximumRemaining = PerfClock::time_point::max() - now;
    m_deadline = timeout > maximumRemaining ? PerfClock::time_point::max() : now + timeout;
}

bool Deadline::IsExpired() const noexcept
{
    return PerfClock::now() >= m_deadline;
}

PerfClock::duration Deadline::Remaining() const noexcept
{
    const PerfClock::time_point now = PerfClock::now();
    return now < m_deadline ? m_deadline - now : PerfClock::duration::zero();
}

struct tm* LocalTime(const time_t* timer, struct tm* buffer)
{
    if (timer == nullptr || buffer == nullptr)
    {
        return nullptr;
    }

#if defined(_WIN32)
    return localtime_s(buffer, timer) == 0 ? buffer : nullptr;
#else
    return localtime_r(timer, buffer);
#endif
}

std::string GetTimeStringUTC(std::chrono::system_clock::time_point time)
{
    const TimeParts parts = GetTimeParts(time);
    struct tm utcTime{};
    return UniversalTime(&parts.timer, &utcTime) != nullptr
               ? FormatTime(utcTime, parts.milliseconds, "Z")
               : std::string{};
}

std::string GetTimeStringUTC()
{
    return GetTimeStringUTC(std::chrono::system_clock::now());
}

std::string GetLocalTimeStringISO8601(std::chrono::system_clock::time_point time)
{
    const TimeParts parts = GetTimeParts(time);
    struct tm localTime{};
    struct tm utcTime{};
    if (LocalTime(&parts.timer, &localTime) == nullptr ||
        UniversalTime(&parts.timer, &utcTime) == nullptr)
    {
        return {};
    }

    const std::string offset = GetUtcOffset(localTime, utcTime);
    return !offset.empty() ? FormatTime(localTime, parts.milliseconds, offset) : std::string{};
}

std::string GetLocalTimeStringISO8601()
{
    return GetLocalTimeStringISO8601(std::chrono::system_clock::now());
}

std::optional<std::chrono::system_clock::time_point> ParseTimeISO8601(std::string_view value)
{
    int year;
    int month;
    int day;
    int hour;
    int minute;
    int second;
    if (value.size() < 20 || value[4] != '-' || value[7] != '-' || value[10] != 'T' ||
        value[13] != ':' || value[16] != ':' || !ParseNumber(value, 0, 4, year) ||
        !ParseNumber(value, 5, 2, month) || !ParseNumber(value, 8, 2, day) ||
        !ParseNumber(value, 11, 2, hour) || !ParseNumber(value, 14, 2, minute) ||
        !ParseNumber(value, 17, 2, second))
    {
        return std::nullopt;
    }

    std::size_t pos = 19;
    std::int64_t fractionalNanoseconds = 0;
    if (pos < value.size() && value[pos] == '.')
    {
        const std::size_t fractionStart = ++pos;
        while (pos < value.size() && value[pos] >= '0' && value[pos] <= '9')
        {
            if (pos - fractionStart == 9)
            {
                return std::nullopt;
            }
            fractionalNanoseconds = fractionalNanoseconds * 10 + value[pos] - '0';
            ++pos;
        }

        const std::size_t fractionLength = pos - fractionStart;
        if (fractionLength == 0)
        {
            return std::nullopt;
        }
        for (std::size_t i = fractionLength; i < 9; ++i)
        {
            fractionalNanoseconds *= 10;
        }
    }

    int offsetMinutes = 0;
    if (pos < value.size() && value[pos] == 'Z')
    {
        ++pos;
    }
    else
    {
        int offsetHour;
        int offsetMinute;
        if (pos + 6 > value.size() || (value[pos] != '+' && value[pos] != '-') ||
            value[pos + 3] != ':' || !ParseNumber(value, pos + 1, 2, offsetHour) ||
            !ParseNumber(value, pos + 4, 2, offsetMinute) || offsetHour > 14 || offsetMinute > 59 ||
            (offsetHour == 14 && offsetMinute != 0))
        {
            return std::nullopt;
        }

        offsetMinutes = offsetHour * 60 + offsetMinute;
        if (value[pos] == '-')
        {
            offsetMinutes = -offsetMinutes;
        }
        pos += 6;
    }

    const std::chrono::year_month_day date{
        std::chrono::year{year},
        std::chrono::month{static_cast<unsigned int>(month)},
        std::chrono::day{static_cast<unsigned int>(day)},
    };
    if (pos != value.size() || !date.ok() || hour > 23 || minute > 59 || second > 59)
    {
        return std::nullopt;
    }

    const std::chrono::sys_seconds utcTime =
        std::chrono::sys_days{date} + std::chrono::hours{hour} +
        std::chrono::minutes{minute - offsetMinutes} + std::chrono::seconds{second};

    using ClockDuration = std::chrono::system_clock::duration;
    const auto fraction = std::chrono::nanoseconds{fractionalNanoseconds};
    const long double wholeTicks =
        std::chrono::duration<long double, ClockDuration::period>{utcTime.time_since_epoch()}
            .count();
    const long double fractionTicks =
        std::chrono::duration<long double, ClockDuration::period>{fraction}.count();
    const long double totalTicks = wholeTicks + fractionTicks;
    if (totalTicks < static_cast<long double>(ClockDuration::min().count()) ||
        totalTicks > static_cast<long double>(ClockDuration::max().count()))
    {
        return std::nullopt;
    }

    return std::chrono::system_clock::time_point{
        std::chrono::duration_cast<ClockDuration>(utcTime.time_since_epoch()) +
        std::chrono::duration_cast<ClockDuration>(fraction)};
}

std::int64_t GetUnixTimeMilliseconds()
{
    return static_cast<std::int64_t>(
        std::chrono::floor<std::chrono::milliseconds>(std::chrono::system_clock::now())
            .time_since_epoch()
            .count());
}

std::optional<std::chrono::system_clock::time_point> FromUnixTimeMilliseconds(std::int64_t value)
{
    using ClockDuration = std::chrono::system_clock::duration;
    if constexpr (std::ratio_less_equal_v<ClockDuration::period, Milliseconds::period>)
    {
        const auto minimum = std::chrono::ceil<Milliseconds>(ClockDuration::min()).count();
        const auto maximum = std::chrono::floor<Milliseconds>(ClockDuration::max()).count();
        if (value < minimum || value > maximum)
        {
            return std::nullopt;
        }
    }

    return std::chrono::system_clock::time_point{
        std::chrono::duration_cast<ClockDuration>(Milliseconds{value})};
}

} // namespace rad
