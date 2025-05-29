#pragma once

#include <SDFramework/Gui/Window.h>
#include <SDFramework/Gui/VulkanContext.h>

#include <vkpp/Core/Instance.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Surface.h>
#include <vkpp/Core/Swapchain.h>

namespace sdf
{

class VulkanWindow : public Window
{
public:
    VulkanWindow();
    ~VulkanWindow();

    virtual bool Create(std::string_view title, int w, int h, SDL_WindowFlags flags) override;
    virtual void Destroy() override;

    static std::set<std::string> GetVulkanInstanceExtensionsRequired();
    rad::Ref<vkpp::Instance> CreateVulkanInstance(std::string_view appName, uint32_t appVersion);
    vkpp::Surface* GetVulkanSurface() const { return m_surface.get(); }
    bool RecreateVulkanSurface();
    // @param gpuIndex: index of the GPU to use, -1 for the default GPU (prefer the first discrete one).
    rad::Ref<vkpp::Device> CreateVulkanDevice(int& gpuIndex);

    virtual bool OnEvent(const SDL_Event& event) override;
    virtual void OnResized(int width, int height) override;

protected:
    rad::Ref<vkpp::Surface> CreateVulkanSurface();

    rad::Ref<vkpp::Instance> m_instance;
    rad::Ref<vkpp::Device> m_device;
    rad::Ref<vkpp::Surface> m_surface;

    rad::Ref<VulkanContext> m_vulkan;

}; // class VulkanWindow

} // namespace sdf
