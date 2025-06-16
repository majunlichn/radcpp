#pragma once

#include <SDFramework/Gui/Window.h>
#include <SDFramework/Gui/VulkanFrame.h>

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
    bool CreateVulkanSurface();
    vkpp::Surface* GetVulkanSurface() const { return m_surface.get(); }

    virtual bool OnEvent(const SDL_Event& event) override;
    virtual void OnResized(int width, int height) override;

protected:

    rad::Ref<vkpp::Instance> m_instance;
    rad::Ref<vkpp::Device> m_device;
    rad::Ref<vkpp::Surface> m_surface;

    rad::Ref<VulkanFrame> m_frame;

}; // class VulkanWindow

} // namespace sdf
