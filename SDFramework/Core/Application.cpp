#include <SDFramework/Core/Application.h>

namespace sdf
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
    Destroy();
}

bool Application::Init(int argc, char** argv)
{
    m_status = Status::Init;
    rad::Application::Init(argc, argv);
    // Init basic subsystems, init others with SDL_InitSubSystem later if required.
    bool result = SDL_Init(
        SDL_INIT_AUDIO | SDL_INIT_VIDEO);
    if (result)
    {
        int version = SDL_GetVersion();
        SDF_LOG(info, "SDL initialized on {}: {}.{}.{} ({})", SDL_GetPlatform(),
            SDL_VERSIONNUM_MAJOR(version), SDL_VERSIONNUM_MINOR(version), SDL_VERSIONNUM_MICRO(version),
            SDL_GetRevision());
        if (const char* pWorkingDir = SDL_GetBasePath())
        {
            SDF_LOG(info, "Working Directory: {}", pWorkingDir);
        }
        if (const char* pVideoDriver = SDL_GetCurrentVideoDriver())
        {
            SDF_LOG(info, "Current Video Driver: {}", pVideoDriver);
        }
        if (const char* pAudioDriver = SDL_GetCurrentAudioDriver())
        {
            SDF_LOG(info, "Current Audio Driver: {}", pAudioDriver);
        }
        UpdateDisplayInfos();
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_Init failed: {}", SDL_GetError());
        return false;
    }
}

void Application::Destroy()
{
    if (g_appInstance)
    {
        SDL_Quit();
        SDF_LOG(info, "SDL quited.");
        g_appInstance = nullptr;
    }
}

SDL_InitFlags Application::GetSubsystemInitialized() const
{
    // Returns a mask of all initialized subsystems if flags is 0.
    return SDL_WasInit(0);
}

bool Application::IsSubsystemInitialized(SDL_InitFlags flags)
{
    SDL_InitFlags mask = SDL_WasInit(flags);
    return rad::HasBits(flags, mask);
}

bool Application::SetMetadataProperty(std::string_view name, std::string_view value)
{
    bool result = SDL_SetAppMetadataProperty(name.data(), value.data());
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetAppMetadataProperty({}, {}) failed: {}",
            name, value, SDL_GetError());
        return false;
    }
}

const char* Application::GetMetadataProperty(std::string_view name)
{
    return SDL_GetAppMetadataProperty(name.data());
}

bool Application::IsMainThread()
{
    return SDL_IsMainThread();
}

bool Application::RunOnMainThread(SDL_MainThreadCallback callback, void* userData, bool waitComplete)
{
    return SDL_RunOnMainThread(callback, userData, waitComplete);
}

void Application::RegisterEventHandler(EventHandler* handler)
{
    std::lock_guard<std::mutex> lock(m_eventMutex);
    m_eventHandlers.push_back(handler);
}

void Application::UnregisterEventHandler(EventHandler* handler)
{
    std::lock_guard<std::mutex> lock(m_eventMutex);
    std::erase(m_eventHandlers, handler);
}

bool Application::PushEvent(SDL_Event& event)
{
    bool result = SDL_PushEvent(&event);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_PushEvent failed: {}", SDL_GetError());
        return false;
    }
}

void Application::OnEvent(const SDL_Event& event)
{
    m_status = Status::Running;
    for (EventHandler* handler : m_eventHandlers)
    {
        if (handler->OnEvent(event))
        {
            return;
        }
    }

    switch (event.type)
    {
    case SDL_EVENT_QUIT: // User-requested quit.
        m_status = Status::Exit;
        break;
    case SDL_EVENT_TERMINATING:
        SDF_LOG(info, "SDL_EVENT_TERMINATING: "
            "The application is being terminated by the OS.");
        m_status = Status::Exit;
        break;
    case SDL_EVENT_LOW_MEMORY:
        SDF_LOG(info, "SDL_EVENT_LOW_MEMORY: "
            "The application is low on memory, free memory if possible.");
        break;
    case SDL_EVENT_WILL_ENTER_BACKGROUND:
        SDF_LOG(info, "SDL_EVENT_WILL_ENTER_BACKGROUND: "
            "The application is about to enter the background.");
        break;
    case SDL_EVENT_DID_ENTER_BACKGROUND:
        SDF_LOG(info, "SDL_EVENT_DID_ENTER_BACKGROUND: "
            "The application did enter the background and may not get CPU for some time.");
        break;
    case SDL_EVENT_WILL_ENTER_FOREGROUND:
        SDF_LOG(info, "SDL_EVENT_WILL_ENTER_FOREGROUND: "
            "The application is about to enter the foreground.");
        break;
    case SDL_EVENT_DID_ENTER_FOREGROUND:
        SDF_LOG(info, "SDL_EVENT_DID_ENTER_FOREGROUND: "
            "The application is now interactive.");
        break;
    case SDL_EVENT_LOCALE_CHANGED:
        SDF_LOG(info, "SDL_EVENT_LOCALE_CHANGED: "
            "The user's locale preferences have changed.");
        break;
    case SDL_EVENT_SYSTEM_THEME_CHANGED:
        SDF_LOG(info, "SDL_EVENT_SYSTEM_THEME_CHANGED: "
            "The system theme changed.");
        break;
    case SDL_EVENT_DISPLAY_ORIENTATION:
    case SDL_EVENT_DISPLAY_ADDED:
    case SDL_EVENT_DISPLAY_REMOVED:
    case SDL_EVENT_DISPLAY_MOVED:
    case SDL_EVENT_DISPLAY_CONTENT_SCALE_CHANGED:
        SDF_LOG(info, "Display state changed.");
        UpdateDisplayInfos();
        break;
    }
}

void Application::OnIdle()
{
    for (EventHandler* handler : m_eventHandlers)
    {
        handler->OnIdle();
    }

    if (m_eventHandlers.empty())
    {
        m_status = Status::Exit;
    }
}

void Application::Exit(int errCode)
{
    m_errCode = errCode;
    m_status = Status::Exit;
}

bool Application::IsScreenSaverEnabled()
{
    return (SDL_ScreenSaverEnabled() == true);
}

bool Application::EnableScreenSaver()
{
    bool result = SDL_EnableScreenSaver();
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_EnableScreenSaver failed: {}", SDL_GetError());
        return false;
    }
}

bool Application::DisableScreenSaver()
{
    bool result = SDL_DisableScreenSaver();
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_DisableScreenSaver failed: {}", SDL_GetError());
        return false;
    }
}

bool Application::SetClipboardText(const char* text)
{
    bool result = SDL_SetClipboardText(text);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetClipboardText failed: {}", SDL_GetError());
        return false;
    }
}

std::string Application::GetClipboardText()
{
    std::string buffer;
    const char* text = SDL_GetClipboardText();
    if (text)
    {
        buffer = text;
    }
    else
    {
        SDF_LOG(err, "SDL_GetClipboardText failed: {}", SDL_GetError());
    }
    return buffer;
}

bool Application::HasClipboardText()
{
    return (SDL_HasClipboardText() == true);
}

bool Application::SetPrimarySelectionText(const char* text)
{
    bool result = SDL_SetPrimarySelectionText(text);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetPrimarySelectionText failed: {}", SDL_GetError());
        return false;
    }
}

std::string Application::GetPrimarySelectionText()
{
    std::string buffer;
    const char* text = SDL_GetPrimarySelectionText();
    if (text)
    {
        buffer = text;
    }
    else
    {
        SDF_LOG(err, "SDL_GetPrimarySelectionText failed: {}", SDL_GetError());
    }
    return buffer;
}

bool Application::HasPrimarySelectionText()
{
    return (SDL_HasPrimarySelectionText() == true);
}

bool Application::SetClipboardData(SDL_ClipboardDataCallback callback, SDL_ClipboardCleanupCallback cleanup,
    void* userData, const char** mimeTypes, size_t mimeTypeCount)
{
    bool result = SDL_SetClipboardData(callback, cleanup, userData, mimeTypes, mimeTypeCount);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetClipboardData failed: {}", SDL_GetError());
        return false;
    }
}

bool Application::ClearClipboardData()
{
    return (SDL_ClearClipboardData() == true);
}

const void* Application::GetClipboardData(const char* mimeType, size_t* size)
{
    if (const void* data = SDL_GetClipboardData(mimeType, size))
    {
        return data;
    }
    else
    {
        SDF_LOG(err, "SDL_GetClipboardData failed: {}", SDL_GetError());
        return nullptr;
    }
}

bool Application::HasClipboardData(const char* mimeType)
{
    return (SDL_HasClipboardData(mimeType) == true);
}

void Application::UpdateDisplayInfos()
{
    int displayCount = 0;
    const SDL_DisplayID* ids = SDL_GetDisplays(&displayCount);
    if (ids == nullptr)
    {
        SDF_LOG(err, "SDL_GetDisplays failed: {}", SDL_GetError());
        m_displays.clear();
        return;
    }
    m_displays.clear();
    m_displays.resize(displayCount);
    for (int i = 0; i < displayCount; ++i)
    {
        DisplayInfo info = {};
        SDL_DisplayID id = ids[i];
        info.id = id;
        if (const char* pName = SDL_GetDisplayName(id))
        {
            info.name = pName;
        }
        else
        {
            SDF_LOG(err, "SDL_GetDisplayName failed: {}", SDL_GetError());
            info.name = "Unknown";
        }

        if (SDL_GetDisplayBounds(id, &info.bounds) != true)
        {
            SDF_LOG(err, "SDL_GetDisplayBounds failed: {}", SDL_GetError());
        }

        if (SDL_GetDisplayUsableBounds(id, &info.usableBounds) != true)
        {
            SDF_LOG(err, "SDL_GetDisplayUsableBounds failed: {}", SDL_GetError());
        }

        info.naturalOrientation = SDL_GetNaturalDisplayOrientation(id);
        info.currentOrientation = SDL_GetCurrentDisplayOrientation(id);

        info.scale = SDL_GetDisplayContentScale(id);
        if (info.scale == 0.0f)
        {
            SDF_LOG(err, "SDL_GetDisplayContentScale failed: {}", SDL_GetError());
        }

        info.propID = SDL_GetDisplayProperties(id);
        if (info.propID != 0)
        {
            info.hdrEnabled = SDL_GetBooleanProperty(info.propID,
                SDL_PROP_DISPLAY_HDR_ENABLED_BOOLEAN, false);
            info.kmsdrmOrientation = SDL_GetNumberProperty(info.propID,
                SDL_PROP_DISPLAY_KMSDRM_PANEL_ORIENTATION_NUMBER, 0);
        }
        else
        {
            SDF_LOG(err, "SDL_GetDisplayProperties failed: {}", SDL_GetError());
        }

        int count = 0;
        const SDL_DisplayMode* const* modes = SDL_GetFullscreenDisplayModes(id, &count);
        if (modes && (count > 0))
        {
            info.modes.resize(count);
            std::memcpy(info.modes.data(), modes, count * sizeof(modes[0]));
        }

        info.desktopMode = SDL_GetDesktopDisplayMode(id);
        info.currentMode = SDL_GetCurrentDisplayMode(id);

        SDF_LOG(info, "Display#{}: {} ({}x{}@{}Hz, {})",
            i, info.name, info.currentMode->w, info.currentMode->h,
            info.currentMode->refresh_rate,
            SDL_GetPixelFormatName(info.currentMode->format));
    }
}

} // namespace sdf
