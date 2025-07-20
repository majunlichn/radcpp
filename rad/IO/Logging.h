#pragma once

#include <rad/Core/Platform.h>
#include <rad/IO/Format.h>
#include <spdlog/spdlog.h>
// https://fmt.dev/dev/api/
#include <fmt/chrono.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

namespace rad
{

// Not thread safe.
bool InitLogging(spdlog::sink_ptr fileSink);
bool InitLogging(const std::string& filename, bool truncate);

spdlog::logger* GetDefaultLogger();
std::shared_ptr<spdlog::logger> CreateLogger(const std::string& name);

} // namespace rad

#if !defined(RAD_NO_LOGGING)
#   define RAD_LOG(LogLevel, ...) SPDLOG_LOGGER_CALL(rad::GetDefaultLogger(), spdlog::level::LogLevel, __VA_ARGS__)
#else
#   define RAD_LOG(LogLevel, ...)
#endif
