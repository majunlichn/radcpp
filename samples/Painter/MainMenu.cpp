#include "MainMenu.h"

MainMenu::MainMenu(PaintManager* manager, sdf::GuiContext* context) :
    Widget(manager, context)
{
}

MainMenu::~MainMenu()
{
}

void MainMenu::OnIdle()
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("New", "Ctrl+N"))
            {
            }
            if (ImGui::MenuItem("Open", "Ctrl+O"))
            {
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))
        {
            if (ImGui::MenuItem("Undo", "CTRL+Z"))
            {
            }
            if (ImGui::MenuItem("Redo", "CTRL+Y", false, false))
            {
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Cut", "CTRL+X"))
            {
            }
            if (ImGui::MenuItem("Copy", "CTRL+C"))
            {
            }
            if (ImGui::MenuItem("Paste", "CTRL+V"))
            {
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help"))
        {
            if (ImGui::MenuItem("Show Demo Window"))
            {
                m_manager->m_showDemoWindow = true;
            }
            if (ImGui::MenuItem("About Dear ImGui"))
            {
                m_manager->m_showAboutWindow = true;
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}
