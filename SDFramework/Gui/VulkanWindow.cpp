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
        m_surface = CreateVulkanSurface();
        SDF_LOG(info, "VulkanWindow created: {} ({}x{})", title, w, h);
        return true;
    }
    else
    {
        return false;
    }
}

void VulkanWindow::Destroy()
{
    m_device->WaitIdle();
    if (m_vulkan)
    {
        m_vulkan.reset();
    }
    Window::Destroy();
}

std::set<std::string> VulkanWindow::GetVulkanInstanceExtensionsRequired()
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

rad::Ref<vkpp::Instance> VulkanWindow::CreateVulkanInstance(
    std::string_view appName, uint32_t appVersion)
{
    rad::Ref<vkpp::Instance> instance = RAD_NEW vkpp::Instance();
    std::set<std::string> instanceLayers = {};
    std::set<std::string> instanceExtensions = GetVulkanInstanceExtensionsRequired();
    if (instance->Init(
        appName, appVersion,
        appName, appVersion,
        instanceLayers, instanceExtensions))
    {
        return instance;
    }
    else
    {
        return nullptr;
    }
}

rad::Ref<vkpp::Surface> VulkanWindow::CreateVulkanSurface()
{
    VkSurfaceKHR surfaceHandle = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(GetHandle(), m_instance->GetHandle(), nullptr, &surfaceHandle))
    {
        SDF_LOG(err, "SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
    }
    return RAD_NEW vkpp::Surface(m_instance, surfaceHandle);
}

bool VulkanWindow::RecreateVulkanSurface()
{
    m_surface = CreateVulkanSurface();
    return (m_surface != nullptr);
}

rad::Ref<vkpp::Device> VulkanWindow::CreateVulkanDevice(int& gpuIndex)
{
    const auto& physicalDevices = m_instance->m_physicalDevices;
    if (gpuIndex == -1)
    {
        int priorityPrev = 0;
        for (uint32_t i = 0; i < physicalDevices.size(); i++)
        {
            const vk::PhysicalDeviceProperties deviceProps = physicalDevices[i].getProperties();
            assert(deviceProps.deviceType <= vk::PhysicalDeviceType::eCpu);

            auto surfaceSupport = physicalDevices[i].getSurfaceSupportKHR(0, m_surface->GetHandle());
            if (surfaceSupport != vk::True)
            {
                continue;
            }

            std::map<vk::PhysicalDeviceType, int> deviceTypePriorities =
            {
                { vk::PhysicalDeviceType::eDiscreteGpu,     5 },
                { vk::PhysicalDeviceType::eIntegratedGpu,   4 },
                { vk::PhysicalDeviceType::eVirtualGpu,      3 },
                { vk::PhysicalDeviceType::eCpu,             2 },
                { vk::PhysicalDeviceType::eOther,           1 },
            };

            int priority = -1;
            if (deviceTypePriorities.find(deviceProps.deviceType) != deviceTypePriorities.end())
            {
                priority = deviceTypePriorities[deviceProps.deviceType];
            }

            if (priority > priorityPrev)
            {
                gpuIndex = i;
                priorityPrev = priority;
            }
        }
    }
    assert((gpuIndex >= 0) && (gpuIndex < static_cast<int>(physicalDevices.size())));
    vk::raii::PhysicalDevice physicalDevice = physicalDevices[gpuIndex];
    return m_instance->CreateDevice(physicalDevice);
}

bool VulkanWindow::OnEvent(const SDL_Event& event)
{
    if (m_vulkan)
    {
        m_vulkan->ProcessEvent(event);
    }
    return Window::OnEvent(event);
}

void VulkanWindow::OnResized(int width, int height)
{
    m_device->WaitIdle();
    m_vulkan->Resize(width, height);
}

} // namespace sdf
