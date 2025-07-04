#include <rad/System/Application.h>

#include <rad/IO/File.h>
#include <rad/IO/Logging.h>
#include <rad/System/CpuInfo.h>

#include <backward.hpp>

#include <boost/nowide/args.hpp>
#include <boost/nowide/cstdlib.hpp>

namespace rad
{

static Application* g_appInstance = nullptr;

Application* Application::GetInstance()
{
    return g_appInstance;
}

Application::Application()
{
    assert(g_appInstance == nullptr);
    g_appInstance = this;
}

Application::~Application()
{
    g_appInstance = nullptr;
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
    std::string logFileName = pystring::os::path::basename(argv[0]) + ".log";
    FilePath logFilePath = GetCurrentPath() / MakeFilePath(logFileName);
    InitLogging(ToString(logFilePath), true);

    RAD_LOG(info, "radcpp Version: {}.{}.{}",
        RAD_VERSION_MAJOR, RAD_VERSION_MINOR, RAD_VERSION_PATCH);

    RAD_LOG(info, "Program: {}", argv[0]);
    RAD_LOG(info, "Working Dir: {}",
        (const char*)rad::GetWorkingDirectory().u8string().c_str());

#if defined(CPU_FEATURES_ARCH_X86)
    RAD_LOG(info, "CPU: {} ({})",
        rad::StrTrim(rad::g_X86Info.brand_string),
        rad::g_X86Info.vendor);
    RAD_LOG(info, "Number of Physical Cores: {}", rad::GetNumberOfPhysicalCores());
#endif

    return true;
}

// Implement with backward-cpp: https://github.com/bombela/backward-cpp
// C++23 <stacktrace>: https://en.cppreference.com/w/cpp/header/stacktrace
void Application::PrintStackTrace(std::ostream& stream, int depth)
{
    using namespace backward;
    // On Windows, must delcare Printer before load_here, or the first print won't work, is it a bug?
    Printer p;
    StackTrace st;
    p.color_mode = ColorMode::never;
    p.trace_context_size = 9;
    st.skip_n_firsts(2); // skip current and load_here.
    st.load_here(depth);
    p.print(st, stream);
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

} // namespace rad
