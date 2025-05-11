#pragma once

#include <SDFramework/Core/Common.h>
#include <SDL3/SDL_timer.h>

namespace sdf
{

// Get the current value of the high resolution counter.
Uint64 GetPerfCounter();
// Get the frequency in Hz of the high resolution counter.
Uint64 GetPerfCounterFrequency();

// Wait a specified number of milliseconds before returning.
void DelayMS(Uint32 ms);
// Wait a specified number of nanoseconds before returning.
void DelayNS(Uint64 ns);

using TimerCallbackMS = SDL_TimerCallback;
using TimerCallbackNS = SDL_NSTimerCallback;

class Timer : public rad::RefCounted<Timer>
{
public:
    Timer();
    virtual ~Timer();

    bool StartMS(Uint32 interval, TimerCallbackMS callback, void* userData);
    bool StartNS(Uint64 interval, TimerCallbackNS callback, void* userData);
    void Stop();

    SDL_TimerID GetID() const { return m_id; }

protected:
    SDL_TimerID m_id = 0;
}; // class Timer

class TimerTaskMS : public Timer
{
public:
    // SDL_TimerCallback
    virtual Uint32 Callback(SDL_TimerID timerID, Uint32 interval) = 0;
    bool Start(Uint32 interval);

}; // class TimerTaskMS

class TimerTaskNS : public Timer
{
public:
    // SDL_NSTimerCallback
    virtual Uint64 Callback(SDL_TimerID timerID, Uint64 interval) = 0;
    bool Start(Uint64 interval);

}; // class TimerTaskMS

} // namespace sdf
