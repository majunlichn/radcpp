#pragma once

#include <SDFramework/Gui/Renderer.h>

#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_sdlrenderer3.h>

namespace sdf
{

class GuiContext : public rad::RefCounted<GuiContext>
{
public:
    GuiContext(Window* window, Renderer* renderer);
    ~GuiContext();

    bool Init();
    void Destroy();
    bool ProcessEvent(const SDL_Event& event);

    void NewFrame();
    void Render();

private:
    Window* m_window;
    Renderer* m_renderer;

    ImGuiContext* m_context = nullptr;

}; // class GuiContext

} // namespace sdf
