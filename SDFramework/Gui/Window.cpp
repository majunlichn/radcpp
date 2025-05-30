#include <SDFramework/Gui/Window.h>
#include <SDFramework/Gui/Surface.h>

namespace sdf
{

Window::Window()
{
}

Window::~Window()
{
    Destroy();
}

bool Window::Create(std::string_view title, int w, int h, SDL_WindowFlags flags)
{
    m_handle = SDL_CreateWindow(title.data(), w, h, flags);
    if (m_handle)
    {
        m_id = SDL_GetWindowID(m_handle);
        if (m_id != 0)
        {
            Application::GetInstance()->RegisterEventHandler(this);
            return true;
        }
        else
        {
            SDF_LOG(err, "SDL_GetWindowID failed: {}", SDL_GetError());
            return false;
        }
    }
    else
    {
        SDF_LOG(err, "SDL_CreateWindow failed: {}", SDL_GetError());
        return false;
    }
}

void Window::Destroy()
{
    if (m_handle)
    {
        Application::GetInstance()->UnregisterEventHandler(this);
        SDL_DestroyWindow(m_handle);
        m_handle = nullptr;
    }
}

float Window::GetPixelDensity()
{
    float density = SDL_GetWindowPixelDensity(m_handle);
    if (density == 0)
    {
        SDF_LOG(err, "SDL_GetWindowPixelDensity failed: {}", SDL_GetError());
    }
    return density;
}

float Window::GetDisplayScale()
{
    float scale = SDL_GetWindowDisplayScale(m_handle);
    if (scale == 0)
    {
        SDF_LOG(err, "SDL_GetWindowDisplayScale failed: {}", SDL_GetError());
    }
    return scale;
}

bool Window::SetFullscreenMode(const SDL_DisplayMode* mode)
{
    bool result = SDL_SetWindowFullscreenMode(m_handle, mode);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowFullscreenMode failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetFullscreenMode(const SDL_DisplayMode& mode)
{
    return SetFullscreenMode(&mode);
}

bool Window::SetBorderlessFullscreenDesktopMode()
{
    return SetFullscreenMode(nullptr);
}

const SDL_DisplayMode* Window::GetFullscreenMode()
{
    return SDL_GetWindowFullscreenMode(m_handle);
}

const void* Window::GetICCProfile(size_t* size)
{
    const void* data = SDL_GetWindowICCProfile(m_handle, size);
    if (data == nullptr)
    {
        SDF_LOG(err, "SDL_GetWindowICCProfile failed: {}", SDL_GetError());
    }
    return data;
}

Uint32 Window::GetPixelFormat()
{
    Uint32 format = SDL_GetWindowPixelFormat(m_handle);
    if (format == SDL_PIXELFORMAT_UNKNOWN)
    {
        SDF_LOG(err, "SDL_GetWindowPixelFormat failed: {}", SDL_GetError());
    }
    return format;
}

SDL_WindowFlags Window::GetFlags()
{
    return SDL_GetWindowFlags(m_handle);
}

bool Window::SetTitle(std::string_view title)
{
    bool result = SDL_SetWindowTitle(m_handle, title.data());
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowTitle failed: {}", SDL_GetError());
        return false;
    }
}

const char* Window::GetTitle()
{
    return SDL_GetWindowTitle(m_handle);
}

bool Window::SetIcon(SDL_Surface* icon)
{
    bool result = SDL_SetWindowIcon(m_handle, icon);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowIcon failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetIcon(Surface* icon)
{
    return SetIcon(icon->GetHandle());
}

bool Window::SetPosition(int x, int y)
{
    bool result = SDL_SetWindowPosition(m_handle, x, y);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowPosition failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::GetPosition(int* x, int* y)
{
    bool result = SDL_GetWindowPosition(m_handle, x, y);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetWindowPosition failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetSize(int w, int h)
{
    bool result = SDL_SetWindowSize(m_handle, w, h);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowSize failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::Resize(int w, int h)
{
    return SetSize(w, h);
}

bool Window::GetSize(int* w, int* h)
{
    bool result = SDL_GetWindowSize(m_handle, w, h);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetWindowSize failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetAspectRatio(float minAspect, float maxAspect)
{
    bool result = SDL_SetWindowAspectRatio(m_handle, minAspect, maxAspect);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowAspectRatio failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::GetAspectRatio(float* minAspect, float* maxAspect)
{
    bool result = SDL_GetWindowAspectRatio(m_handle, minAspect, maxAspect);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetWindowAspectRatio failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::GetSizeInPixels(int* w, int* h)
{
    bool result = SDL_GetWindowSizeInPixels(m_handle, w, h);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetWindowSizeInPixels failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetMinimumSize(int w, int h)
{
    bool result = SDL_SetWindowMinimumSize(m_handle, w, h);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowMinimumSize failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::GetMinimumSize(int* w, int* h)
{
    bool result = SDL_GetWindowMinimumSize(m_handle, w, h);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetWindowMinimumSize failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetMaximumSize(int w, int h)
{
    bool result = SDL_SetWindowMaximumSize(m_handle, w, h);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowMaximumSize failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::GetMaximumSize(int* w, int* h)
{
    bool result = SDL_GetWindowMaximumSize(m_handle, w, h);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetWindowMaximumSize failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetBordered(bool bordered)
{
    bool result = SDL_SetWindowBordered(m_handle,
        bordered ? true : false);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowBordered failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetResizable(bool resizable)
{
    bool result = SDL_SetWindowResizable(m_handle, resizable);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowResizable failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetAlwaysOnTop(bool onTop)
{
    bool result = SDL_SetWindowAlwaysOnTop(m_handle,
        onTop ? true : false);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowAlwaysOnTop failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::Show()
{
    bool result = SDL_ShowWindow(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_ShowWindow failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::Hide()
{
    bool result = SDL_HideWindow(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_HideWindow failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::Raise()
{
    bool result = SDL_RaiseWindow(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RaiseWindow failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::Maximize()
{
    bool result = SDL_MaximizeWindow(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_MaximizeWindow failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::Minimize()
{
    bool result = SDL_MinimizeWindow(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_MinimizeWindow failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::Restore()
{
    bool result = SDL_RestoreWindow(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_RestoreWindow failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetFullscreen(bool fullscreen)
{
    bool result = SDL_SetWindowFullscreen(m_handle, fullscreen);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowFullscreen failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::Sync()
{
    bool result = SDL_SyncWindow(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SyncWindow timed out!");
        return false;
    }
}

bool Window::HasSurface()
{
    return (SDL_WindowHasSurface(m_handle) == true);
}

SDL_Surface* Window::GetSurface()
{
    SDL_Surface* surface = SDL_GetWindowSurface(m_handle);
    if (surface == nullptr)
    {
        SDF_LOG(err, "SDL_GetWindowSurface failed: {}", SDL_GetError());
    }
    return surface;
}

bool Window::SetSurfaceVSync(int vsync)
{
    bool result = SDL_SetWindowSurfaceVSync(m_handle, vsync);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowSurfaceVSync failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::GetSurfaceVSync(int* vsync)
{
    bool result = SDL_GetWindowSurfaceVSync(m_handle, vsync);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetWindowSurfaceVSync failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::UpdateSurface()
{
    bool result = SDL_UpdateWindowSurface(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_UpdateWindowSurface failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::UpdateSurfaceRects(rad::Span<SDL_Rect> rects)
{
    bool result = SDL_UpdateWindowSurfaceRects(m_handle,
        rects.data(), static_cast<int>(rects.size()));
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_UpdateWindowSurfaceRects failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::DestroySurface()
{
    bool result = SDL_DestroyWindowSurface(m_handle);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_DestroyWindowSurface failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetKeyboardGrab(bool grabbed)
{
    bool result = SDL_SetWindowKeyboardGrab(m_handle, grabbed ? true : false);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowKeyboardGrab failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetMouseGrab(bool grabbed)
{
    bool result = SDL_SetWindowMouseGrab(m_handle,
        grabbed ? true : false);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowMouseGrab failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::GetKeyboardGrab()
{
    return (SDL_GetWindowKeyboardGrab(m_handle) == true);
}

bool Window::GetMouseGrab()
{
    return (SDL_GetWindowMouseGrab(m_handle) == true);
}

bool Window::SetMouseRect(const SDL_Rect* rect)
{
    bool result = SDL_SetWindowMouseRect(m_handle, rect);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowMouseRect failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetMouseRect(const SDL_Rect& rect)
{
    return SetMouseRect(&rect);
}

const SDL_Rect* Window::GetMouseRect()
{
    return SDL_GetWindowMouseRect(m_handle);
}

bool Window::SetOpacity(float opacity)
{
    bool result = SDL_SetWindowOpacity(m_handle, opacity);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowOpacity failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::GetOpacity(float* opacity)
{
    float result = SDL_GetWindowOpacity(m_handle);
    if (result >= 0)
    {
        *opacity = result;
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_GetWindowOpacity failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetFocusable(bool focusable)
{
    bool result = SDL_SetWindowFocusable(m_handle,
        focusable ? true : false);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowFocusable failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::ShowSystemMenu(int x, int y)
{
    bool result = SDL_ShowWindowSystemMenu(m_handle, x, y);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_ShowWindowSystemMenu failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetShape(SDL_Surface* shape)
{
    bool result = SDL_SetWindowShape(m_handle, shape);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_SetWindowShape failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::SetShape(Surface* shape)
{
    return SetShape(shape->GetHandle());
}

bool Window::Flash(SDL_FlashOperation operation)
{
    bool result = SDL_FlashWindow(m_handle, operation);
    if (result)
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "SDL_FlashWindow failed: {}", SDL_GetError());
        return false;
    }
}

bool Window::OnEvent(const SDL_Event& event)
{
    for (EventHandler* handler : m_eventHandlers)
    {
        if (handler->OnEvent(event))
        {
            return true;
        }
    }

    if ((event.type >= SDL_EVENT_WINDOW_FIRST) &&
        (event.type <= SDL_EVENT_WINDOW_LAST))
    {
        if (event.window.windowID == m_id)
        {
            OnWindowEvent(event.window);
            return true;
        }
    }
    else if (event.type == SDL_EVENT_KEY_DOWN)
    {
        if (event.key.windowID == m_id)
        {
            OnKeyDown(event.key);
            return true;
        }
    }
    else if (event.type == SDL_EVENT_KEY_UP)
    {
        if (event.key.windowID == m_id)
        {
            OnKeyUp(event.key);
            return true;
        }
    }
    else if (event.type == SDL_EVENT_TEXT_EDITING)
    {
        if (event.edit.windowID == m_id)
        {
            OnTextEditing(event.edit);
            return true;
        }
    }
    else if (event.type == SDL_EVENT_TEXT_INPUT)
    {
        if (event.text.windowID == m_id)
        {
            OnTextInput(event.text);
            return true;
        }
    }
    else if (event.type == SDL_EVENT_MOUSE_MOTION)
    {
        if (event.motion.windowID == m_id)
        {
            OnMouseMove(event.motion);
            return true;
        }
    }
    else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN)
    {
        if (event.button.windowID == m_id)
        {
            OnMouseButtonDown(event.button);
            return true;
        }
    }
    else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP)
    {
        if (event.button.windowID == m_id)
        {
            OnMouseButtonUp(event.button);
            return true;
        }
    }
    else if (event.type == SDL_EVENT_MOUSE_WHEEL)
    {
        if (event.wheel.windowID == m_id)
        {
            OnMouseWheel(event.wheel);
            return true;
        }
    }
    else if (event.type == SDL_EVENT_USER)
    {
        if ((event.user.windowID == 0) ||
            (event.user.windowID == m_id))
        {
            OnUserEvent(event.user);
            return (event.user.windowID == m_id);
        }
    }

    return false;
}

void Window::OnIdle()
{
}

void Window::OnWindowEvent(const SDL_WindowEvent& event)
{
    switch (event.type)
    {
    case SDL_EVENT_WINDOW_SHOWN:
        OnShown();
        break;
    case SDL_EVENT_WINDOW_HIDDEN:
        OnHidden();
        break;
    case SDL_EVENT_WINDOW_EXPOSED:
        OnExposed();
        break;
    case SDL_EVENT_WINDOW_MOVED:
        OnMoved(event.data1, event.data2);
        break;
    case SDL_EVENT_WINDOW_RESIZED:
        OnResized(event.data1, event.data2);
        break;
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        OnPixelSizeChanged(event.data1, event.data2);
        break;
    case SDL_EVENT_WINDOW_MINIMIZED:
        OnMinimized();
        break;
    case SDL_EVENT_WINDOW_MAXIMIZED:
        OnMaximized();
        break;
    case SDL_EVENT_WINDOW_RESTORED:
        OnRestored();
        break;
    case SDL_EVENT_WINDOW_MOUSE_ENTER:
        OnMouseEnter();
        break;
    case SDL_EVENT_WINDOW_MOUSE_LEAVE:
        OnMouseLeave();
        break;
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
        OnFocusGained();
        break;
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        OnFocusLost();
        break;
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
        OnCloseRequested();
        break;
    case SDL_EVENT_WINDOW_HIT_TEST:
        OnHitTest();
        break;
    case SDL_EVENT_WINDOW_ICCPROF_CHANGED:
        OnIccProfileChanged();
        break;
    case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
        OnDisplayChanged();
        break;
    case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
        OnDisplayScaleChanged();
        break;
    case SDL_EVENT_WINDOW_OCCLUDED:
        OnOccluded();
        break;
    case SDL_EVENT_WINDOW_ENTER_FULLSCREEN:
        OnEnterFullscreen();
        break;
    case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN:
        OnLeaveFullscreen();
        break;
    case SDL_EVENT_WINDOW_DESTROYED:
        OnDestroyed();
        break;
    }
}

} // namespace sdf
