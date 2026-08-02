#include <rad/System/Application.h>

#include <rad/Core/Memory.h>
#include <rad/Core/Platform.h>
#include <rad/Diagnostics/Exception.h>
#include <rad/Diagnostics/StackTrace.h>
#include <rad/IO/Logging.h>
#include <rad/System/OS.h>

#include <backward.hpp>
#include <boost/nowide/args.hpp>
#include <boost/nowide/filesystem.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <clocale>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if defined(RAD_OS_WINDOWS)
#include <windows.h>
#if defined(RAD_COMPILER_MSVC) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#else
#include <unistd.h>
#endif

namespace rad
{
namespace
{

bool IsUtf8Locale(std::string_view name)
{
    std::string normalized(name);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char character)
                   { return static_cast<char>(std::tolower(character)); });
    return normalized.find("utf-8") != std::string::npos ||
           normalized.find("utf8") != std::string::npos;
}

[[nodiscard]] const char* AllocationKindName(AllocationKind kind) noexcept
{
    switch (kind)
    {
    case AllocationKind::Unknown:
        return "Unknown";
    case AllocationKind::Raw:
        return "Raw";
    case AllocationKind::RawAligned:
        return "RawAligned";
    case AllocationKind::Object:
        return "Object";
    case AllocationKind::ObjectArray:
        return "ObjectArray";
    }
    return "Unknown";
}

#if RAD_ENABLE_MEMORY_TRACKING
void ReportRemainingAllocations()
{
    try
    {
        MemoryTracker& tracker = GetGlobalMemoryTracker();
        const MemoryStatistics statistics = tracker.Statistics();
        if (statistics.activeAllocationCount == 0)
        {
            return;
        }

        const auto allocations = tracker.ActiveAllocations();
        std::ostringstream message;
        message << "Detected " << statistics.activeAllocationCount << " active allocation(s), "
                << statistics.activeBytes << " byte(s) still tracked at exit:\n";
        for (const auto& [address, record] : allocations)
        {
            static_cast<void>(address);
            message << "  " << record.address << " size=" << record.size
                    << " kind=" << AllocationKindName(record.kind) << " at "
                    << record.location.file_name() << ':' << record.location.line() << " ("
                    << record.location.function_name() << ')';
            if (!record.stackTrace.empty())
            {
                message << "\n" << record.stackTrace;
            }
            message << '\n';
        }

        const std::string text = message.str();
        if (spdlog::logger* logger = Application::Instance().GetLogger())
        {
            logger->warn("{}", text);
            logger->flush();
        }
        else
        {
            std::cerr << text << std::flush;
        }
    }
    catch (...)
    {
    }
}

#endif // RAD_ENABLE_MEMORY_TRACKING

} // namespace

Application& Application::Instance()
{
    static Application app;
    return app;
}

void Application::DefaultTerminateHandler() noexcept
{
    try
    {
        std::string reason = "Unhandled exception";
        bool hasCapturedStackTrace = false;
        if (const std::exception_ptr exception = std::current_exception())
        {
            try
            {
                std::rethrow_exception(exception);
            }
            catch (const Exception& error)
            {
                reason += ":\n";
                reason += error.DiagnosticInformation();
                hasCapturedStackTrace = !error.StackTrace().empty();
            }
            catch (const std::exception& error)
            {
                reason += ": ";
                reason += error.what();
            }
            catch (...)
            {
                reason += " of unknown type";
            }
        }
        else
        {
            reason = "std::terminate called without an active exception";
        }

        if (!hasCapturedStackTrace)
        {
            reason += '\n';
            reason += GetStackTrace();
        }

        auto& app = Instance();
        if (app.m_logger)
        {
            app.m_logger->critical("{}", reason);
            app.m_logger->flush();
        }
        else
        {
            std::cerr << reason << std::endl;
        }
    }
    catch (...)
    {
        constexpr std::string_view fallback = "Application terminated unexpectedly.\n";
#if defined(RAD_OS_WINDOWS)
        const HANDLE errorHandle = GetStdHandle(STD_ERROR_HANDLE);
        if (errorHandle != nullptr && errorHandle != INVALID_HANDLE_VALUE)
        {
            DWORD written = 0;
            WriteFile(errorHandle, fallback.data(), static_cast<DWORD>(fallback.size()), &written,
                      nullptr);
        }
#else
        const auto ignored = ::write(STDERR_FILENO, fallback.data(), fallback.size());
        static_cast<void>(ignored);
#endif
    }

    std::abort();
}

void Application::Init(int argc, char** argv)
{
    InitArguments(argc, argv);

    ConfigureUtf8Locale();
    ConfigureUtf8Console();
    EnableDebugFeatures();

    InitLogging();
#if RAD_ENABLE_MEMORY_TRACKING
    static const bool remainingAllocationReporterRegistered =
        (std::atexit(ReportRemainingAllocations) == 0);
    static_cast<void>(remainingAllocationReporterRegistered);
#endif
    InstallDefaultSignalHandlers();
    InstallDefaultTerminateHandler();

    if (auto* logger = GetLogger())
    {
        logger->info("Executable: {}", PathToUtf8(os::executable_path()));
#if defined(_DEBUG)
        const auto& args = Arguments();
        for (std::size_t index = 1; index < args.size(); ++index)
        {
            logger->debug("Argument[{}]: {}", index, args[index]);
        }
#endif
        logger->info("Working directory: {}", PathToUtf8(os::getcwd()));
        logger->info("Temporary directory: {}", PathToUtf8(os::temp_directory_path()));
    }
}

const std::vector<std::string>& Application::Arguments() const noexcept
{
    return m_arguments;
}

spdlog::logger* Application::GetLogger() const noexcept
{
    return m_logger.get();
}

void Application::InitLogging()
{
    if (m_logger)
    {
        return;
    }

    auto& logManager = LogManager::Instance();
    const os::FilePath executablePath = os::executable_path();
    const bool logInitialized = logManager.IsInitialized();
    if (!logInitialized)
    {
        os::FilePath logName = executablePath.stem();
        logName += os::FilePath{".log"};
        logManager.Init(PathToUtf8(os::getcwd() / logName), true);
    }

    try
    {
        m_logger = logManager.CreateLogger(PathToUtf8(executablePath.stem()));
    }
    catch (...)
    {
        if (!logInitialized)
        {
            logManager.Shutdown();
        }
        throw;
    }
}

void Application::InstallDefaultSignalHandlers()
{
    if (m_signalHandlersInstalled)
    {
        return;
    }

    static backward::SignalHandling signalHandling;
    if (!signalHandling.loaded())
    {
        if (m_logger)
        {
            m_logger->warn("Fatal signal diagnostics are unavailable on this platform");
        }
    }
    m_signalHandlersInstalled = true;
    if (m_logger)
    {
        m_logger->debug("Default signal handlers installed");
    }
}

void Application::InstallDefaultTerminateHandler() noexcept
{
    if (m_terminateHandlerInstalled)
    {
        return;
    }

    std::set_terminate(DefaultTerminateHandler);
    m_terminateHandlerInstalled = true;
    if (m_logger)
    {
        m_logger->debug("Default terminate handler installed");
    }
}

void Application::Exit(int code)
{
    try
    {
        if (auto* logger = Instance().GetLogger())
        {
            logger->info("Exiting with code {}", code);
        }
    }
    catch (...)
    {
    }
    std::exit(code);
}

void Application::QuickExit(int code) noexcept
{
    try
    {
        if (auto* logger = Instance().GetLogger())
        {
            logger->warn("Quick exit requested with code {}", code);
        }
        auto& logManager = LogManager::Instance();
        if (logManager.IsInitialized())
        {
            logManager.Flush();
        }
    }
    catch (...)
    {
    }
    std::quick_exit(code);
}

void Application::Terminate() noexcept
{
    std::terminate();
}

void Application::Abort() noexcept
{
    try
    {
        if (auto* logger = Instance().GetLogger())
        {
            logger->critical("Abort requested");
            logger->flush();
        }
    }
    catch (...)
    {
    }
    std::abort();
}

void Application::InitArguments(int argc, char** argv)
{
    if (argc < 0 || (argc > 0 && argv == nullptr))
    {
        throw std::invalid_argument{"Invalid command-line arguments"};
    }

    int utf8Argc = argc;
    char** utf8Argv = argv;
    boost::nowide::args convertedArguments(utf8Argc, utf8Argv);

    m_arguments.clear();
    m_arguments.reserve(static_cast<std::size_t>(utf8Argc));
    for (int index = 0; index < utf8Argc; ++index)
    {
        if (utf8Argv[index] == nullptr)
        {
            throw std::invalid_argument{"Command-line argument must not be null"};
        }
        m_arguments.emplace_back(utf8Argv[index]);
    }
}

void Application::ConfigureUtf8Locale()
{
#if defined(RAD_OS_WINDOWS)
    constexpr const char* candidates[] = {".UTF-8"};
#else
    constexpr const char* candidates[] = {"C.UTF-8", "en_US.UTF-8", "UTF-8", ""};
#endif

    const char* configuredLocale = nullptr;
    for (const char* candidate : candidates)
    {
        configuredLocale = std::setlocale(LC_ALL, candidate);
        if (configuredLocale != nullptr && IsUtf8Locale(configuredLocale))
        {
            break;
        }
        configuredLocale = nullptr;
    }
    if (configuredLocale == nullptr)
    {
        throw std::runtime_error{"No UTF-8 locale is available"};
    }

    const std::locale locale(configuredLocale);
    std::locale::global(locale);
    std::cin.imbue(locale);
    std::cout.imbue(locale);
    std::cerr.imbue(locale);
    std::clog.imbue(locale);
    static_cast<void>(boost::nowide::nowide_filesystem());
}

void Application::ConfigureUtf8Console()
{
#if defined(RAD_OS_WINDOWS)
    if (GetConsoleCP() != 0 && SetConsoleCP(CP_UTF8) == 0)
    {
        throw std::system_error(
            std::error_code(static_cast<int>(GetLastError()), std::system_category()),
            "SetConsoleCP");
    }
    if (GetConsoleOutputCP() != 0 && SetConsoleOutputCP(CP_UTF8) == 0)
    {
        throw std::system_error(
            std::error_code(static_cast<int>(GetLastError()), std::system_category()),
            "SetConsoleOutputCP");
    }
#endif
}

void Application::EnableDebugFeatures() noexcept
{
    EnableMemoryLeakDetection();
}

// Enables automatic process-exit leak reporting on MSVC Debug; no-op elsewhere.
void Application::EnableMemoryLeakDetection() noexcept
{
#if defined(RAD_OS_WINDOWS) && defined(RAD_COMPILER_MSVC) && defined(_DEBUG)
    int oldFlags = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
    _CrtSetDbgFlag(oldFlags & ~_CRTDBG_ALLOC_MEM_DF);
    // Workaround false positives:
    try
    {
        (void)std::chrono::current_zone();
    }
    catch (...)
    {
        // Ignore failures
    }
    // Track subsequent allocations and dump surviving blocks automatically at process exit.
    _CrtSetDbgFlag(oldFlags | _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
}

} // namespace rad
