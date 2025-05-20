#pragma once

#include <SDFramework/Gui/Window.h>
#include <vkpp/Core/Instance.h>
#include <vkpp/Core/Surface.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Swapchain.h>

namespace sdf
{

class VulkanWindow : public Window
{
public:
    VulkanWindow(rad::Ref<vkpp::Instance> instance);
    ~VulkanWindow();

    virtual bool Create(const char* title, int w, int h, SDL_WindowFlags flags) override;
    virtual void Destroy() override;

    vkpp::Surface* GetSurface() const { return m_surface.get(); }
    vkpp::Swapchain* GetSwapchain() const { return m_swapchain.get(); }

private:
    rad::Ref<vkpp::Surface> CreateSurface();
    rad::Ref<vkpp::Swapchain> CreateSwapchain(int width, int height);

    rad::Ref<vkpp::Instance> m_instance;
    rad::Ref<vkpp::Surface> m_surface;
    rad::Ref<vkpp::Device> m_device;

    vk::SurfaceCapabilitiesKHR m_surfaceCaps;
    std::vector<vk::SurfaceFormatKHR> m_surfaceFormats;
    std::vector<vk::PresentModeKHR> m_presentModes;
    vk::SurfaceFormatKHR m_surfaceFormat = {};
    vk::PresentModeKHR m_presentMode = vk::PresentModeKHR::eFifo;
    rad::Ref<vkpp::Swapchain> m_swapchain;

}; // class VulkanWindow

} // namespace sdf
