#pragma once

#include <rad/Core/Platform.h>
#include <rad/Core/Integer.h>
#include <rad/Core/Float.h>
#include <rad/Core/Flags.h>
#include <rad/Core/Memory.h>
#include <rad/Core/RefCounted.h>
#include <rad/Core/TypeTraits.h>
#include <rad/Container/SmallVector.h>
#include <rad/Container/Span.h>
#include <rad/IO/File.h>
#include <rad/IO/Logging.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_log.h>

namespace sdf
{

spdlog::logger* GetLogger();

#ifndef SDF_NO_LOGGING
// LogLevel: trace, debug, info, warn, err, critical
#define SDF_LOG(LogLevel, ...) SPDLOG_LOGGER_CALL(sdf::GetLogger(), spdlog::level::LogLevel, __VA_ARGS__)
#else
#define SDF_LOG(LogLevel, ...)
#endif

void SetOutOfMemory();

// Buffer that is allocated by SDL but needs to be freed by user.
class SDL_Buffer
{
public:
    SDL_Buffer(Uint8* buffer) : m_data(buffer) {}
    ~SDL_Buffer();
    Uint8* m_data = nullptr;
}; // class SDL_Buffer

// Prefer triple-buffering.
static constexpr uint32_t MaxSwapchainImageCount = 3;

// Allow a maximum of two outstanding presentation operations.
static constexpr uint32_t MaxFrameLag = 2;

} // namespace sdf
