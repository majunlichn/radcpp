#include "VulkanWindow.h"
#include <SDL3/SDL_vulkan.h>

namespace sdf
{

VulkanWindow::VulkanWindow()
{
}

VulkanWindow::~VulkanWindow()
{
}

bool VulkanWindow::Create(std::string_view title, int w, int h, SDL_WindowFlags flags)
{
    if (Window::Create(title.data(), w, h, flags | SDL_WINDOW_VULKAN))
    {
        if (CreateVulkanSurface())
        {
            SDF_LOG(info, "VulkanWindow created: {} ({}x{})", title, w, h);
            return true;
        }
    }
    return false;
}

void VulkanWindow::Destroy()
{
    m_device->WaitIdle();
    if (m_frame)
    {
        m_frame.reset();
    }
    Window::Destroy();
}

std::set<std::string> VulkanWindow::GetRequiredVulkanInstanceExtensions()
{
    Uint32 count = 0;
    const char* const* ppExtensionNames = SDL_Vulkan_GetInstanceExtensions(&count);
    std::set<std::string> extensionNames;
    for (Uint32 i = 0; i < count; ++i)
    {
        extensionNames.insert(ppExtensionNames[i]);
    }
    return extensionNames;
}

bool VulkanWindow::CreateVulkanSurface()
{
    VkSurfaceKHR surfaceHandle = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(GetHandle(), m_instance->GetHandle(), nullptr, &surfaceHandle))
    {
        SDF_LOG(err, "SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
    }
    m_surface = RAD_NEW vkpp::Surface(m_instance, surfaceHandle);
    return (m_surface != nullptr);
}

bool VulkanWindow::OnEvent(const SDL_Event& event)
{
    if (m_frame)
    {
        m_frame->ProcessEvent(event);
    }
    return Window::OnEvent(event);
}

void VulkanWindow::OnResized(int width, int height)
{
    m_device->WaitIdle();
    m_frame->Resize(width, height);
}

} // namespace sdf
