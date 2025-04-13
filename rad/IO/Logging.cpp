#include <rad/IO/Logging.h>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/msvc_sink.h>
#include <spdlog/cfg/env.h>

namespace rad
{

static std::shared_ptr<spdlog::logger> DefaultLogger;

spdlog::logger* GetDefaultLogger()
{
    return DefaultLogger.get();
}

bool InitLogging(spdlog::sink_ptr fileSink)
{
    DefaultLogger = spdlog::stdout_color_mt("Default");
    spdlog::set_default_logger(DefaultLogger);

    if (fileSink)
    {
        DefaultLogger->sinks().push_back(fileSink);
    }

#if defined(RAD_COMPILER_MSC) && defined(_DEBUG)
    spdlog::sink_ptr msvcSink = std::make_shared<spdlog::sinks::msvc_sink_mt>();
    DefaultLogger->sinks().push_back(msvcSink);
#endif

    // https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
    spdlog::set_pattern("%^[%T.%e] %n (%l)%$: %v");
    spdlog::flush_on(spdlog::level::warn);
    spdlog::cfg::load_env_levels();

    return DefaultLogger.get();
}

bool InitLogging(const std::string& filename, bool truncate)
{
    spdlog::sink_ptr fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, truncate);
    return InitLogging(fileSink);
}

std::shared_ptr<spdlog::logger> CreateLogger(const std::string& name)
{
    auto& sinks = DefaultLogger->sinks();
    std::shared_ptr<spdlog::logger> logger = std::make_shared<spdlog::logger>(
        name, sinks.begin(), sinks.end());
    spdlog::initialize_logger(logger);
    return logger;
}

} // namespace rad
