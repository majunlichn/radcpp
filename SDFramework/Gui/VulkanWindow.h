#pragma once

#include <SDFramework/Gui/Window.h>
#include <vkpp/Core/Instance.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Surface.h>
#include <vkpp/Core/Swapchain.h>

namespace sdf
{

class VulkanWindow : public Window
{
public:
    VulkanWindow(rad::Ref<vkpp::Instance> instance);
    ~VulkanWindow();

    virtual bool Create(std::string_view title, int w, int h, SDL_WindowFlags flags) override;
    virtual void Destroy() override;

    vkpp::Surface* GetSurface() const { return m_surface.get(); }
    vkpp::Swapchain* GetSwapchain() const { return m_swapchain.get(); }

private:
    rad::Ref<vkpp::Surface> CreateSurface();
    rad::Ref<vkpp::Swapchain> CreateSwapchain(int width, int height);

    rad::Ref<vkpp::Instance> m_instance;
    rad::Ref<vkpp::Surface> m_surface;
    rad::Ref<vkpp::Device> m_device;

    rad::Ref<vkpp::Swapchain> m_swapchain;
    vk::PresentModeKHR m_presentMode = vk::PresentModeKHR::eFifo;
    bool m_isMinimized = false;
    bool m_isFirstSwapchainFrame = false;


}; // class VulkanWindow

} // namespace sdf
