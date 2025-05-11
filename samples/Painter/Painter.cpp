#include "Painter.h"

Painter::Painter()
{
    m_logger = rad::CreateLogger("Painter");
    m_logger->trace(__FUNCTION__);
}

Painter::~Painter()
{
    m_logger->trace(__FUNCTION__);
}

bool Painter::Init()
{
    SDL_WindowFlags flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN;
    Create("Painter", 1600, 900, flags);

    m_renderer = RAD_NEW sdf::Renderer(this);
    if (!m_renderer->Init())
    {
        return false;
    }
    m_renderer->SetVSync(1);

    m_gui = RAD_NEW sdf::GuiContext(this, m_renderer.get());
    if (!m_gui->Init())
    {
        return false;
    }
    return true;
}

bool Painter::OnEvent(const SDL_Event& event)
{
    if (m_gui)
    {
        m_gui->ProcessEvent(event);
    }
    return Window::OnEvent(event);
}

void Painter::OnIdle()
{
    if (GetFlags() & SDL_WINDOW_MINIMIZED)
    {
        return;
    }
    m_renderer->Clear();
    m_gui->NewFrame();
    if (m_showDemoWindow)
    {
        ImGui::ShowDemoWindow(&m_showDemoWindow);
    }
    m_gui->Render();
    m_renderer->Present();
}

void Painter::OnShown()
{
    m_logger->trace(__FUNCTION__);
}

void Painter::OnHidden()
{
    m_logger->trace(__FUNCTION__);
}

void Painter::OnExposed()
{
    m_logger->trace(__FUNCTION__);
}

void Painter::OnMoved(int x, int y)
{
    m_logger->trace("OnMoved: {:4}, {:4}", x, y);
}

void Painter::OnResized(int width, int height)
{
    m_logger->trace("OnResized: {:4}, {:4}", width, height);
}

void Painter::OnPixelSizeChanged(int width, int height)
{
    m_logger->trace("OnPixelSizeChanged: {:4}, {:4}", width, height);
}

void Painter::OnMinimized()
{
    m_logger->trace("OnMinimized");
}

void Painter::OnMaximized()
{
    m_logger->trace("OnMaximized");
}

void Painter::OnRestored()
{
    m_logger->trace("OnRestored");
}

void Painter::OnMouseEnter()
{
    m_logger->trace("OnMouseEnter");
}

void Painter::OnMouseLeave()
{
    m_logger->trace("OnMouseLeave");
}

void Painter::OnFocusGained()
{
    m_logger->trace("OnFocusGained");
}

void Painter::OnFocusLost()
{
    m_logger->trace("OnFocusLost");
}

void Painter::OnCloseRequested()
{
    m_logger->trace("OnCloseRequested");
    Destroy();
}

void Painter::OnHitTest()
{
    m_logger->trace("OnHitTest");
}

void Painter::OnIccProfileChanged()
{
    m_logger->trace("OnIccProfileChanged");
}

void Painter::OnDisplayChanged()
{
    m_logger->trace("OnDisplayChanged");
}

void Painter::OnDisplayScaleChanged()
{
    m_logger->trace("OnDisplayScaleChanged");
}

void Painter::OnOccluded()
{
    m_logger->trace("OnOccluded");
}

void Painter::OnEnterFullscreen()
{
    m_logger->trace("OnEnterFullscreen");
}

void Painter::OnLeaveFullscreen()
{
    m_logger->trace("OnLeaveFullscreen");
}

void Painter::OnDestroyed()
{
    m_logger->trace("OnDestroyed");
}

void Painter::OnKeyDown(const SDL_KeyboardEvent& keyDown)
{
    m_logger->trace("OnKeyDown: {}", SDL_GetKeyName(keyDown.key));
    if (keyDown.key == SDLK_F1)
    {
        m_showDemoWindow = !m_showDemoWindow;
    }
}

void Painter::OnKeyUp(const SDL_KeyboardEvent& keyUp)
{
    m_logger->trace("OnKeyUp: {}", SDL_GetKeyName(keyUp.key));
}

void Painter::OnTextEditing(const SDL_TextEditingEvent& textEditing)
{
    m_logger->trace("OnTextEditing: {}", textEditing.text);
}

void Painter::OnTextInput(const SDL_TextInputEvent& textInput)
{
    m_logger->trace("OnTextInput: {}", textInput.text);
}

void Painter::OnMouseMove(const SDL_MouseMotionEvent& mouseMotion)
{
    m_logger->trace("OnMouseMove: x={:4} ({:+4}); y={:4} ({:+4})",
        mouseMotion.x, mouseMotion.xrel, mouseMotion.y, mouseMotion.yrel);
}

void Painter::OnMouseButtonDown(const SDL_MouseButtonEvent& mouseButton)
{
    m_logger->trace("OnMouseButtonDown: {}", GetMouseButtonName(mouseButton.button));
}

void Painter::OnMouseButtonUp(const SDL_MouseButtonEvent& mouseButton)
{
    m_logger->trace("OnMouseButtonUp: {}", GetMouseButtonName(mouseButton.button));
}

void Painter::OnMouseWheel(const SDL_MouseWheelEvent& mouseWheel)
{
    m_logger->trace("OnMouseWheel: {:+}", mouseWheel.y);
}

void Painter::OnUserEvent(const SDL_UserEvent& user)
{
    m_logger->trace("OnUserEvent");
}

const char* Painter::GetMouseButtonName(Uint8 button)
{
    switch (button)
    {
    case SDL_BUTTON_LEFT: return "SDL_BUTTON_LEFT";
    case SDL_BUTTON_MIDDLE: return "SDL_BUTTON_MIDDLE";
    case SDL_BUTTON_RIGHT: return "SDL_BUTTON_RIGHT";
    case SDL_BUTTON_X1: return "SDL_BUTTON_X1";
    case SDL_BUTTON_X2: return "SDL_BUTTON_X2";
    }
    return "SDL_BUTTON_UNKNOWN";
}
