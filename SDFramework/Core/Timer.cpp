#include <SDFramework/Core/Timer.h>

namespace sdf
{

Uint64 GetPerfCounter()
{
    return SDL_GetPerformanceCounter();
}

Uint64 GetPerfCounterFrequency()
{
    return SDL_GetPerformanceFrequency();
}

void DelayMS(Uint32 ms)
{
    SDL_Delay(ms);
}

void DelayNS(Uint64 ns)
{
    SDL_DelayNS(ns);
}

Timer::Timer()
{
}

Timer::~Timer()
{
}

bool Timer::StartMS(Uint32 interval, TimerCallbackMS callback, void* userData)
{
    assert(m_id == 0);
    m_id = SDL_AddTimer(interval, callback, userData);
    if (m_id != 0)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_AddTimer failed: {}", SDL_GetError());
        return false;
    }
}

bool Timer::StartNS(Uint64 interval, TimerCallbackNS callback, void* userData)
{
    assert(m_id == 0);
    m_id = SDL_AddTimerNS(interval, callback, userData);
    if (m_id != 0)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_AddTimerNS failed: {}", SDL_GetError());
        return false;
    }
}

void Timer::Stop()
{
    if (m_id != 0)
    {
        if (SDL_RemoveTimer(m_id) != 0)
        {
            SDF_LOG(err, "SDL_RemoveTimer failed: {}", SDL_GetError());
        }
        m_id = 0;
    }
}

bool TimerTaskMS::Start(Uint32 interval)
{
    return Timer::StartMS(interval, [](void* userData, SDL_TimerID timerID, Uint32 interval)->Uint32 {
        return reinterpret_cast<TimerTaskMS*>(userData)->Callback(timerID, interval);
        }, this);
}

bool TimerTaskNS::Start(Uint64 interval)
{
    return Timer::StartNS(interval, [](void* userData, SDL_TimerID timerID, Uint64 interval)->Uint64 {
        return reinterpret_cast<TimerTaskNS*>(userData)->Callback(timerID, interval);
        }, this);
}

} // namespace sdf
