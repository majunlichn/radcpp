#pragma once

#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rad
{

// LogManager owns process-wide logging configuration for newly created loggers.
// It is not thread-safe; configure it during startup before sharing loggers across threads.
class LogManager
{
public:
    static LogManager& Instance();

    // fileName is expected to be UTF-8 encoded when provided.
    void Init(std::string_view fileName = {}, bool truncate = false);
    [[nodiscard]] bool IsInitialized() const noexcept;
    // Logger names are registered with spdlog and must be unique until Shutdown().
    [[nodiscard]] std::shared_ptr<spdlog::logger> CreateLogger(std::string_view name);

    void AddSink(spdlog::sink_ptr sink);
    // fileName is expected to be UTF-8 encoded.
    void AddFileSink(std::string_view fileName, bool truncate = false);
    void ClearSinks();

    void SetLevel(spdlog::level::level_enum level);
    void SetFlushLevel(spdlog::level::level_enum level);
    void SetPattern(std::string pattern);

    void Flush();
    void Shutdown();

private:
    LogManager() = default;

    std::vector<spdlog::sink_ptr> m_sinks;
}; // class LogManager

} // namespace rad
