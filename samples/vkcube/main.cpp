#define SDL_MAIN_USE_CALLBACKS 1

#include "CubeDemo.h"

#include <SDFramework/Core/Application.h>
#include <SDL3/SDL_main.h>

rad::Ref<sdf::Application> g_app;
rad::Ref<CubeDemo> g_demo;

SDL_AppResult SDL_AppInit(void** appState, int argc, char** argv)
{
    g_app = RAD_NEW sdf::Application();
    if (g_app->Init(argc, argv))
    {
        *reinterpret_cast<sdf::Application**>(appState) = g_app.get();
        g_app->SetMetadataProperty(SDL_PROP_APP_METADATA_NAME_STRING, "Painter");
    }
    else
    {
        return SDL_APP_FAILURE;
    }
    g_demo = RAD_NEW CubeDemo();
    if (!g_demo->Init(argc, argv))
    {
        return SDL_APP_FAILURE;
    }
    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appState)
{
    sdf::Application* app = reinterpret_cast<sdf::Application*>(appState);
    assert(g_app == app);

    g_app->OnIdle();
    if (g_app->GetStatus() == sdf::Application::Status::Running)
    {
        return SDL_APP_CONTINUE;
    }
    else
    {
        if (g_app->GetErrorCode() != 0)
        {
            return SDL_APP_FAILURE;
        }
        else
        {
            return SDL_APP_SUCCESS;
        }
    }
}

SDL_AppResult SDL_AppEvent(void* appState, SDL_Event* event)
{
    sdf::Application* app = reinterpret_cast<sdf::Application*>(appState);
    assert(g_app == app);

    g_app->OnEvent(*event);
    if (g_app->GetStatus() == sdf::Application::Status::Running)
    {
        return SDL_APP_CONTINUE;
    }
    else
    {
        if (g_app->GetErrorCode() != 0)
        {
            return SDL_APP_FAILURE;
        }
        else
        {
            return SDL_APP_SUCCESS;
        }
    }
}

void SDL_AppQuit(void* appState, SDL_AppResult result)
{
    sdf::Application* app = reinterpret_cast<sdf::Application*>(appState);
    assert(g_app == app);

    g_demo.reset();
    g_app.reset();
}
