#pragma once

#include <rad/Core/Platform.h>
#include <spdlog/spdlog.h>
// https://fmt.dev/dev/api/
#include <fmt/ranges.h>
#include <fmt/chrono.h>
#include <fmt/std.h>

namespace rad
{

spdlog::logger* GetDefaultLogger();

// Not thread safe.
bool InitLogging(spdlog::sink_ptr fileSink);
bool InitLogging(const std::string& filename, bool truncate);
std::shared_ptr<spdlog::logger> CreateLogger(const std::string& name);

} // namespace rad

#if !defined(RAD_NO_LOGGING)
#define RAD_LOG(Logger, Level, ...) SPDLOG_LOGGER_CALL(Logger, spdlog::level::Level, __VA_ARGS__)
#define RAD_LOG_DEFAULT(Level, ...) SPDLOG_LOGGER_CALL(rad::GetDefaultLogger(), spdlog::level::Level, __VA_ARGS__)
#else
#define RAD_LOG(Logger, Level, ...)
#define RAD_LOG_DEFAULT(Level, ...)
#endif
