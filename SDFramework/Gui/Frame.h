#pragma once

#include <SDFramework/Gui/Renderer.h>

#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_sdlrenderer3.h>
#include <implot/implot.h>

namespace sdf
{

class Frame : public rad::RefCounted<Frame>
{
public:
    Frame(Window* window, rad::Ref<Renderer> renderer);
    ~Frame();

    bool Init();
    void Destroy();
    bool ProcessEvent(const SDL_Event& event);

    void Render();

    void BeginFrame();
    void EndFrame();

private:
    Window* m_window;
    rad::Ref<Renderer> m_renderer;

    ImGuiContext* m_gui = nullptr;
    ImPlotContext* m_plot = nullptr;

}; // class Frame

} // namespace sdf
