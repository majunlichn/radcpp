#pragma once

#include <rad/IO/Logging.h>

#include <memory>
#include <string>
#include <vector>

namespace rad
{

// Configures process-wide application behavior. Call Init once during single-threaded startup.
class Application final
{
public:
    [[nodiscard]] static Application& Instance();

    void Init(int argc, char** argv);

    // Arguments are UTF-8 encoded and remain valid for the process lifetime.
    [[nodiscard]] const std::vector<std::string>& Arguments() const noexcept;
    [[nodiscard]] spdlog::logger* GetLogger() const noexcept;

    void InstallDefaultSignalHandlers();
    void InstallDefaultTerminateHandler() noexcept;

    [[noreturn]] static void Exit(int code = 0);
    [[noreturn]] static void QuickExit(int code = 0) noexcept;
    [[noreturn]] static void Terminate() noexcept;
    [[noreturn]] static void Abort() noexcept;

private:
    Application() = default;

    static void DefaultTerminateHandler() noexcept;
    void InitArguments(int argc, char** argv);
    void InitLogging();
    static void ConfigureUtf8Locale();
    static void ConfigureUtf8Console();
    static void EnableDebugFeatures() noexcept;
    static void EnableMemoryLeakDetection() noexcept;

    bool m_signalHandlersInstalled = false;
    bool m_terminateHandlerInstalled = false;
    std::vector<std::string> m_arguments;
    std::shared_ptr<spdlog::logger> m_logger;
}; // class Application

} // namespace rad
