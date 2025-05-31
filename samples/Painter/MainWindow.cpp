#include "MainWindow.h"
#include "MainMenu.h"

MainWindow::MainWindow()
{
    m_logger = rad::CreateLogger("MainWindow");
    m_logger->trace(__FUNCTION__);
}

MainWindow::~MainWindow()
{
    m_logger->trace(__FUNCTION__);
}

bool MainWindow::Init()
{
    SDL_WindowFlags flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN;
    Create("Painter", 1600, 900, flags);

    m_manager = RAD_NEW PaintManager();

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

    m_mainMenu = RAD_NEW MainMenu(m_manager.get(), m_gui.get());

    return true;
}

bool MainWindow::OnEvent(const SDL_Event& event)
{
    if (m_gui)
    {
        m_gui->ProcessEvent(event);
    }
    return Window::OnEvent(event);
}

void MainWindow::OnIdle()
{
    if (GetFlags() & SDL_WINDOW_MINIMIZED)
    {
        return;
    }
    m_renderer->Clear();
    m_gui->NewFrame();

    if (m_mainMenu && m_mainMenu->IsEnabled())
    {
        m_mainMenu->OnIdle();
    }

    if (m_manager->m_showDemoWindow)
    {
        ImGui::ShowDemoWindow(&m_manager->m_showDemoWindow);
    }

    if (m_manager->m_showPlotDemoWindow)
    {
        ImPlot::ShowDemoWindow(&m_manager->m_showPlotDemoWindow);
    }

    if (m_manager->m_showAboutWindow)
    {
        ImGui::ShowAboutWindow(&m_manager->m_showAboutWindow);
    }

    m_gui->Render();
    m_renderer->Present();
}

void MainWindow::OnShown()
{
    m_logger->trace(__FUNCTION__);
}

void MainWindow::OnHidden()
{
    m_logger->trace(__FUNCTION__);
}

void MainWindow::OnExposed()
{
    m_logger->trace(__FUNCTION__);
}

void MainWindow::OnMoved(int x, int y)
{
    m_logger->trace("OnMoved: {:4}, {:4}", x, y);
}

void MainWindow::OnResized(int width, int height)
{
    m_logger->trace("OnResized: {:4}, {:4}", width, height);
}

void MainWindow::OnPixelSizeChanged(int width, int height)
{
    m_logger->trace("OnPixelSizeChanged: {:4}, {:4}", width, height);
}

void MainWindow::OnMinimized()
{
    m_logger->trace("OnMinimized");
}

void MainWindow::OnMaximized()
{
    m_logger->trace("OnMaximized");
}

void MainWindow::OnRestored()
{
    m_logger->trace("OnRestored");
}

void MainWindow::OnMouseEnter()
{
    m_logger->trace("OnMouseEnter");
}

void MainWindow::OnMouseLeave()
{
    m_logger->trace("OnMouseLeave");
}

void MainWindow::OnFocusGained()
{
    m_logger->trace("OnFocusGained");
}

void MainWindow::OnFocusLost()
{
    m_logger->trace("OnFocusLost");
}

void MainWindow::OnCloseRequested()
{
    m_logger->trace("OnCloseRequested");
    Destroy();
}

void MainWindow::OnHitTest()
{
    m_logger->trace("OnHitTest");
}

void MainWindow::OnIccProfileChanged()
{
    m_logger->trace("OnIccProfileChanged");
}

void MainWindow::OnDisplayChanged()
{
    m_logger->trace("OnDisplayChanged");
}

void MainWindow::OnDisplayScaleChanged()
{
    m_logger->trace("OnDisplayScaleChanged");
}

void MainWindow::OnOccluded()
{
    m_logger->trace("OnOccluded");
}

void MainWindow::OnEnterFullscreen()
{
    m_logger->trace("OnEnterFullscreen");
}

void MainWindow::OnLeaveFullscreen()
{
    m_logger->trace("OnLeaveFullscreen");
}

void MainWindow::OnDestroyed()
{
    m_logger->trace("OnDestroyed");
}

void MainWindow::OnKeyDown(const SDL_KeyboardEvent& keyDown)
{
    m_logger->trace("OnKeyDown: {}", SDL_GetKeyName(keyDown.key));
    if (keyDown.key == SDLK_F1)
    {
        m_manager->m_showDemoWindow = !m_manager->m_showDemoWindow;
    }
    if (keyDown.key == SDLK_F2)
    {
        m_manager->m_showPlotDemoWindow = !m_manager->m_showPlotDemoWindow;
    }
}

void MainWindow::OnKeyUp(const SDL_KeyboardEvent& keyUp)
{
    m_logger->trace("OnKeyUp: {}", SDL_GetKeyName(keyUp.key));
}

void MainWindow::OnTextEditing(const SDL_TextEditingEvent& textEditing)
{
    m_logger->trace("OnTextEditing: {}", textEditing.text);
}

void MainWindow::OnTextInput(const SDL_TextInputEvent& textInput)
{
    m_logger->trace("OnTextInput: {}", textInput.text);
}

void MainWindow::OnMouseMove(const SDL_MouseMotionEvent& mouseMotion)
{
    m_logger->trace("OnMouseMove: x={:4} ({:+4}); y={:4} ({:+4})",
        mouseMotion.x, mouseMotion.xrel, mouseMotion.y, mouseMotion.yrel);
}

void MainWindow::OnMouseButtonDown(const SDL_MouseButtonEvent& mouseButton)
{
    m_logger->trace("OnMouseButtonDown: {}", GetMouseButtonName(mouseButton.button));
}

void MainWindow::OnMouseButtonUp(const SDL_MouseButtonEvent& mouseButton)
{
    m_logger->trace("OnMouseButtonUp: {}", GetMouseButtonName(mouseButton.button));
}

void MainWindow::OnMouseWheel(const SDL_MouseWheelEvent& mouseWheel)
{
    m_logger->trace("OnMouseWheel: {:+}", mouseWheel.y);
}

void MainWindow::OnUserEvent(const SDL_UserEvent& user)
{
    m_logger->trace("OnUserEvent");
}

const char* MainWindow::GetMouseButtonName(Uint8 button)
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
