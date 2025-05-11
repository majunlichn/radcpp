#pragma once

#include <SDFramework/Core/Common.h>
#include <SDFramework/Core/EventHandler.h>

#include <rad/System/Application.h>

#include <atomic>
#include <mutex>

namespace sdf
{

struct DisplayInfo
{
    SDL_DisplayID id;
    const char* name;
    SDL_Rect bounds;
    // This is the same area as bounds, but with portions reserved by the system removed.
    SDL_Rect usableBounds;
    SDL_DisplayOrientation naturalOrientation;
    SDL_DisplayOrientation currentOrientation;
    // The content scale is the expected scale for content based on the DPI settings of the display.
    float scale;

    // https://wiki.libsdl.org/SDL3/SDL_GetDisplayProperties
    SDL_PropertiesID propID;
    bool hdrEnabled;
    float sdrWhitePoint;
    float hdrHeadroom;
    Sint64 kmsdrmOrientation;

    std::vector<const SDL_DisplayMode*> modes;
    const SDL_DisplayMode* desktopMode;
    const SDL_DisplayMode* currentMode;

}; // struct DisplayInfo

// Make sure only one instance of Application exists.
class Application : public rad::Application
{
public:
    enum class Status : int
    {
        Unknown,
        Init,
        Running,
        Exit,
    };

    Application();
    ~Application();

    static Application* GetInstance();

    bool Init(int argc, char** argv);
    void Destroy();

    const std::vector<DisplayInfo>& GetDisplayInfos() { return m_displays; }

    SDL_InitFlags GetSubsystemInitialized() const;
    bool IsSubsystemInitialized(SDL_InitFlags flags);

    bool SetMetadataProperty(std::string_view name, std::string_view value);
    const char* GetMetadataProperty(std::string_view name);

    bool IsMainThread();
    bool RunOnMainThread(SDL_MainThreadCallback callback, void* userData, bool waitComplete);

    void RegisterEventHandler(EventHandler* handler);
    void UnregisterEventHandler(EventHandler* handler);
    // Return true on success; false if the event is filtered or on failure (event queue being full).
    bool PushEvent(SDL_Event& event);
    void OnEvent(const SDL_Event& event);
    void OnIdle();

    void SetStatus(Status status) { m_status = status; }
    Status GetStatus() { return m_status; }
    void SetErrorCode(int errCode) { m_errCode = errCode; }
    int GetErrorCode() const { return m_errCode; }

    void Exit(int errCode);

    bool IsScreenSaverEnabled();
    bool EnableScreenSaver();
    bool DisableScreenSaver();

    // Put UTF-8 text into the clipboard.
    bool SetClipboardText(const char* text);
    std::string GetClipboardText();
    bool HasClipboardText();

    // Put UTF-8 text into the primary selection.
    bool SetPrimarySelectionText(const char* text);
    std::string GetPrimarySelectionText();
    bool HasPrimarySelectionText();

    // Tell the operating system that the application is offering clipboard data
    // for each of the proivded mime types.
    bool SetClipboardData(SDL_ClipboardDataCallback callback, SDL_ClipboardCleanupCallback cleanup,
        void* userData, const char** mimeTypes, size_t mimeTypeCount);
    bool ClearClipboardData();
    const void* GetClipboardData(const char* mimeType, size_t* size);
    bool HasClipboardData(const char* mimeType);

private:
    std::mutex m_eventMutex;
    std::vector<EventHandler*> m_eventHandlers;

    std::atomic<Status> m_status = Status::Unknown;
    std::atomic_int m_errCode = 0;

    std::vector<DisplayInfo> m_displays;
    void UpdateDisplayInfos();

}; // class Application

} // namespace sdf
