#pragma once

#include <radcpp/Core/Platform.h>
#include <radcpp/Core/RefCounted.h>
#include <radcpp/Core/String.h>

namespace rad
{

class Application : public RefCounted<Application>
{
public:
    Application();
    virtual ~Application();

    bool Init(int argc, char** argv);
    int GetArgc() const { return m_argc; }
    // Retrieves UTF-16 arguments from the Windows API, converts them to UTF-8.
    const std::vector<std::string>& GetArgv() { return m_argv; }
    const char* GetArgv(size_t index) { return m_argv[index].c_str(); }

    void PrintStackTrace(int depth = 32);

    // Causes normal program termination to occur.
    // Functions registered with std::atexit are called in the reverse order of their registration.
    void Exit(int code);
    // Causes normal program termination to occur without completely cleaning the resources.
    // Functions passed to std::at_quick_exit are called in reverse order of their registration.
    void QuickExit(int code);
    // In any case, std::terminate calls the currently installed std::terminate_handler.
    // The default std::terminate_handler calls std::abort.
    void Terminate();
    // Causes abnormal program termination (without cleaning up)
    // unless SIGABRT is being caught by a signal handler.
    void Abort();

private:
    bool InstallDefaultSignalHandlers();
    bool LogSystemInfo();

    int m_argc = 0;
    std::vector<std::string> m_argv;

}; // Application

Application* GetApp();

} // namespace rad
