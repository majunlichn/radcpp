#include <rad/IO/Logging.h>

#include <rad/Core/Platform.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/cfg/env.h>

#if defined(RAD_COMPILER_MSVC)
#include <spdlog/sinks/msvc_sink.h>
#endif

#include <stdexcept>
#include <utility>

namespace rad
{

LogManager& LogManager::Instance()
{
    static LogManager manager;
    return manager;
}

void LogManager::Init(std::string_view fileName, bool truncate)
{
    m_sinks.clear();
    spdlog::set_level(spdlog::level::info);
    spdlog::flush_on(spdlog::level::warn);
    spdlog::set_pattern("%^[%T.%e] %n (%l)%$: %v");
    spdlog::cfg::load_env_levels();

    m_sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
#if defined(RAD_COMPILER_MSVC) && defined(_DEBUG)
    m_sinks.push_back(std::make_shared<spdlog::sinks::msvc_sink_mt>());
#endif
    if (!fileName.empty())
    {
        m_sinks.push_back(
            std::make_shared<spdlog::sinks::basic_file_sink_mt>(std::string{fileName}, truncate));
    }
}

bool LogManager::IsInitialized() const noexcept
{
    return !m_sinks.empty();
}

std::shared_ptr<spdlog::logger> LogManager::CreateLogger(std::string_view name)
{
    if (m_sinks.empty())
    {
        throw std::logic_error{"logging has no sinks; call Init or AddSink before CreateLogger"};
    }

    auto logger =
        std::make_shared<spdlog::logger>(std::string{name}, m_sinks.begin(), m_sinks.end());
    spdlog::initialize_logger(logger);
    return logger;
}

void LogManager::AddSink(spdlog::sink_ptr sink)
{
    if (!sink)
    {
        throw std::invalid_argument{"sink must not be null"};
    }

    m_sinks.push_back(std::move(sink));
}

void LogManager::AddFileSink(std::string_view fileName, bool truncate)
{
    AddSink(std::make_shared<spdlog::sinks::basic_file_sink_mt>(std::string{fileName}, truncate));
}

void LogManager::ClearSinks()
{
    m_sinks.clear();
}

void LogManager::SetLevel(spdlog::level::level_enum level)
{
    spdlog::set_level(level);
}

void LogManager::SetFlushLevel(spdlog::level::level_enum level)
{
    spdlog::flush_on(level);
}

void LogManager::SetPattern(std::string pattern)
{
    spdlog::set_pattern(std::move(pattern));
}

void LogManager::Flush()
{
    spdlog::apply_all([](const std::shared_ptr<spdlog::logger>& logger) { logger->flush(); });
}

void LogManager::Shutdown()
{
    m_sinks.clear();
    spdlog::shutdown();
}

} // namespace rad
