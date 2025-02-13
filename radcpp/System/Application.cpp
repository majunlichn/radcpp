#include <radcpp/System/Application.h>
#include <radcpp/System/CpuInfo.h>
#include <radcpp/System/OS.h>
#include <radcpp/IO/FileSystem.h>
#include <radcpp/IO/Logging.h>
#include <backward.hpp>

#include <boost/nowide/args.hpp>
#include <boost/nowide/cstdlib.hpp>

namespace rad
{

static Application* g_app = nullptr;

Application* GetApp()
{
    return g_app;
}

Application::Application()
{
    assert(g_app == nullptr);
    g_app = this;
}

Application::~Application()
{
    g_app = nullptr;
}

bool Application::Init(int argc, char** argv)
{
#if defined(RAD_OS_WINDOWS)
#if defined(_DEBUG)
    ::_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
    boost::nowide::args nowideArgs(argc, argv);
    m_argc = argc;
    for (int i = 0; i < argc; ++i)
    {
        m_argv.resize(argc);
        m_argv[i] = argv[i];
    }
    std::setlocale(LC_ALL, ".UTF-8");
    ::SetConsoleCP(CP_UTF8);
    ::SetConsoleOutputCP(CP_UTF8);
#endif

    InstallDefaultSignalHandlers();
    InitLogging(path::GetFileName(argv[0]) + ".log", true);

    LOG_DEFAULT(info, "Program: {}", argv[0]);
    LOG_DEFAULT(info, "Working Dir: {}",
        (const char*)rad::GetWorkingDirectory().u8string().c_str());

    LogSystemInfo();

    return true;
}

// Implement with backward-cpp: https://github.com/bombela/backward-cpp
// C++23 <stacktrace>: https://en.cppreference.com/w/cpp/header/stacktrace
void Application::PrintStackTrace(int depth)
{
    using namespace backward;
    // On Windows, must delcare Printer before load_here, or the first print won't work, is it a bug?
    Printer p;
    StackTrace st;
    p.color_mode = ColorMode::never;
    p.trace_context_size = 9;
    st.skip_n_firsts(2); // skip current and load_here.
    st.load_here(depth);
    p.print(st);
}

void Application::Exit(int code)
{
    std::exit(code);
}

void Application::QuickExit(int code)
{
    std::quick_exit(code);
}

void Application::Terminate()
{
    std::terminate();
}

void Application::Abort()
{
    std::abort();
}

bool Application::InstallDefaultSignalHandlers()
{
    backward::SignalHandling signalHandling;
    return signalHandling.loaded();
}

bool Application::LogSystemInfo()
{
#if defined(CPU_FEATURES_ARCH_X86)
    SPDLOG_INFO("CPU: {} ({})",
        rad::StrTrim(rad::g_X86Info.brand_string),
        rad::g_X86Info.vendor);
#endif
    return true;
}

} // namespace rad
