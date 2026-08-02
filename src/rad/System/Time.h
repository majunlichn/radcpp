#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>
#include <optional>
#include <string>
#include <string_view>

namespace rad
{

using Nanoseconds = std::chrono::nanoseconds;
using Microseconds = std::chrono::microseconds;
using Milliseconds = std::chrono::milliseconds;
using Seconds = std::chrono::seconds;
using Minutes = std::chrono::minutes;
using Hours = std::chrono::hours;
using Days = std::chrono::days;
using Weeks = std::chrono::weeks;
// Average fixed-duration units, not calendar-aware periods.
// Do not use these aliases for calendar date arithmetic.
using Months = std::chrono::months;
using Years = std::chrono::years;

// Monotonic clock for measuring elapsed time.
using PerfClock = std::chrono::steady_clock;

// Accumulating, non-thread-safe stopwatch. Start resumes after Stop.
class Stopwatch
{
public:
    void Start() noexcept;
    void Stop() noexcept;
    [[nodiscard]] bool IsRunning() const noexcept;

    template <typename Duration = Seconds>
    [[nodiscard]] Duration Elapsed() const
    {
        PerfClock::duration elapsed = m_elapsed;
        if (m_running)
        {
            elapsed += PerfClock::now() - m_start;
        }
        return std::chrono::duration_cast<Duration>(elapsed);
    }

    [[nodiscard]] Milliseconds ElapsedMilliseconds() const;

private:
    PerfClock::time_point m_start{};
    PerfClock::duration m_elapsed{};
    bool m_running = false;
}; // class Stopwatch

// Monotonic timeout deadline.
class Deadline
{
public:
    explicit Deadline(PerfClock::duration timeout) noexcept;

    [[nodiscard]] bool IsExpired() const noexcept;
    [[nodiscard]] PerfClock::duration Remaining() const noexcept;

private:
    PerfClock::time_point m_deadline;
}; // class Deadline

// Cross-platform, thread-safe alternative to std::localtime.
struct tm* LocalTime(const time_t* timer, struct tm* buffer);

// Returns "YYYY-MM-DDThh:mm:ss.sssZ", or an empty string on failure.
std::string GetTimeStringUTC(std::chrono::system_clock::time_point time);
std::string GetTimeStringUTC();

// Returns local time as "YYYY-MM-DDThh:mm:ss.sss±ZZ:ZZ", or an empty string on failure.
std::string GetLocalTimeStringISO8601(std::chrono::system_clock::time_point time);
std::string GetLocalTimeStringISO8601();

// Parses "YYYY-MM-DDThh:mm:ss[.fffffffff](Z|±hh:mm)".
std::optional<std::chrono::system_clock::time_point> ParseTimeISO8601(std::string_view value);

std::int64_t GetUnixTimeMilliseconds();
// Returns empty when value is outside system_clock's representable range.
std::optional<std::chrono::system_clock::time_point> FromUnixTimeMilliseconds(std::int64_t value);

} // namespace rad
