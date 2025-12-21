#pragma once

#include <rad/Common/Platform.h>
#include <rad/Common/TypeTraits.h>

#include <cassert>
#include <cstdint>
#include <ctime>

#include <chrono>

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
using Months = std::chrono::months;
using Years = std::chrono::years;

using SteadyClock = std::conditional_t<
    std::chrono::high_resolution_clock::is_steady,
    std::chrono::high_resolution_clock,
    std::chrono::steady_clock>;

class Stopwatch
{
public:
    using Clock = SteadyClock;
    enum class State
    {
        Running,
        Stopped,
    };
    Stopwatch() {}
    ~Stopwatch() {}

    auto GetFrequency() const { return Clock::period::den; }
    auto IsHighResolution() const { return std::is_same_v<Clock, std::chrono::high_resolution_clock>; }

    void Start()
    {
        assert(m_state == State::Stopped);
        m_state = State::Running;
        m_start = Clock::now();
    }

    void Stop()
    {
        auto curr = Clock::now();
        m_elapsed += curr - m_start;
        assert(m_state == State::Running);
        m_state = State::Stopped;
    }

    void Reset()
    {
        m_elapsed = Clock::duration::zero();
        m_state = State::Stopped;
    }

    void Restart()
    {
        Reset();
        Start();
    }

    bool IsRunning() const { return (m_state == State::Running); }

    template<typename Resolution = Nanoseconds>
    uint64_t GetElapsedTicks() const
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<Resolution>(m_elapsed).count());
    }

    template<typename Resolution = Nanoseconds>
    double GetElapsedSeconds() const
    {
        uint64_t ticks = GetElapsedTicks<Resolution>();
        return double(ticks) / 1e9;
    }

    template<typename Resolution = Nanoseconds>
    double GetElapsedMilliseconds() const
    {
        uint64_t ticks = GetElapsedTicks<Resolution>();
        return double(ticks) / 1e6;
    }

private:
    State m_state = State::Stopped;
    Clock::time_point m_start;
    Clock::duration m_elapsed{ 0 };

}; // class Stopwatch

struct tm* LocalTime(const time_t* timer, struct tm* buffer);
// Returns string in format "YYYY-MM-DDThh:mm:ssZ" or empty if failed.
std::string GetTimeStringUTC();
// Returns string in format "YYYY-MM-DDThh:mm:ss+0000" or empty if failed.
std::string GetTimeStringISO8601();

} // namespace rad
