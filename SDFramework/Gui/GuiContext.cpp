#include <SDFramework/Gui/GuiContext.h>

namespace sdf
{

GuiContext::GuiContext(Window* window, Renderer* renderer) :
    m_window(window),
    m_renderer(renderer)
{
}

GuiContext::~GuiContext()
{
    Destroy();
}

bool GuiContext::Init()
{
    IMGUI_CHECKVERSION();
    m_gui = ImGui::CreateContext();
    m_plot = ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;   // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;    // Enable Gamepad Controls

    ImGui::StyleColorsDark();

    // Setup scaling:
    ImGuiStyle& style = ImGui::GetStyle();
    float mainScale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    style.ScaleAllSizes(mainScale);
    style.FontScaleDpi = mainScale;

    // Setup Platform/Renderer backends:
    bool result = ImGui_ImplSDL3_InitForSDLRenderer(m_window->GetHandle(), m_renderer->GetHandle());
    if (!result)
    {
        SDF_LOG(err, "ImGui_ImplSDL3_InitForSDLRenderer failed!");
        return false;
    }
    result = ImGui_ImplSDLRenderer3_Init(m_renderer->GetHandle());
    if (!result)
    {
        SDF_LOG(err, "ImGui_ImplSDLRenderer3_Init failed!");
        return false;
    }

    SDL_DisplayID displayID = SDL_GetDisplayForWindow(m_window->GetHandle());
    SDL_Rect rect = {};
    float fontSize = 16.0f;
#if defined(_WIN32)
    auto fonts = ImGui::GetIO().Fonts;
    fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\consola.ttf", fontSize);
#endif
    return true;
}

void GuiContext::Destroy()
{
    if (m_plot)
    {
        ImPlot::DestroyContext(m_plot);
        m_plot = nullptr;
    }
    if (m_gui)
    {
        ImGui_ImplSDLRenderer3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        m_gui = nullptr;
    }
}

bool GuiContext::ProcessEvent(const SDL_Event& event)
{
    return ImGui_ImplSDL3_ProcessEvent(&event);
}

void GuiContext::NewFrame()
{
    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void GuiContext::Render()
{
    ImGuiIO& io = ImGui::GetIO();
    ImGui::Render();
    SDL_SetRenderScale(m_renderer->GetHandle(), io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
    m_renderer->Clear();
    ImGui_ImplSDLRenderer3_RenderDrawData(
        ImGui::GetDrawData(), m_renderer->GetHandle());
}

} // namespace sdf
