#pragma once

#include <rad/IO/Logging.h>

namespace ML
{

spdlog::logger* GetLogger();

} // namespace ML

#define ML_LOG(LogLevel, ...) SPDLOG_LOGGER_CALL(ML::GetLogger(), spdlog::level::LogLevel, __VA_ARGS__)
