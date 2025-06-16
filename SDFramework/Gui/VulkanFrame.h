#pragma once

#include <SDFramework/Core/Common.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Command.h>
#include <vkpp/Core/Fence.h>
#include <vkpp/Core/Semaphore.h>
#include <vkpp/Core/Buffer.h>
#include <vkpp/Core/Image.h>
#include <vkpp/Core/Sampler.h>
#include <vkpp/Core/Descriptor.h>
#include <vkpp/Core/Pipeline.h>

#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>
#include <implot/implot.h>

namespace sdf
{

class VulkanWindow;

class VulkanFrame : public rad::RefCounted<VulkanFrame>
{
public:
    VulkanFrame(VulkanWindow* window, rad::Ref<vkpp::Device> device);
    ~VulkanFrame();

    bool Init();
    void Destroy();

    void Resize(uint32_t width, uint32_t height);

    rad::Ref<vkpp::Swapchain> CreateSwapchain(uint32_t width, uint32_t height);
    vkpp::Swapchain* GetSwapchain() const { return m_swapchain.get(); }
    vkpp::Image* GetRenderTarget() { return m_renderTarget.get(); }
    vkpp::ImageView* GetRenderTargetView() { return m_renderTargetView.get(); }

    bool ProcessEvent(const SDL_Event& event);

    void BeginFrame();
    void Render();
    void EndFrame();

    VulkanWindow* m_window;
    rad::Ref<vkpp::Device> m_device;

    ImGuiContext* m_gui = nullptr;
    ImPlotContext* m_plot = nullptr;

    rad::Ref<vkpp::Swapchain> m_swapchain;
    vk::PresentModeKHR m_presentMode = vk::PresentModeKHR::eFifo;

    rad::Ref<vkpp::Image> m_renderTarget;
    rad::Ref<vkpp::ImageView> m_renderTargetView;
    rad::Ref<vkpp::Image> m_overlay;
    rad::Ref<vkpp::ImageView> m_overlayView;
    rad::Ref<vkpp::Sampler> m_samplerNearest;
    rad::Ref<vkpp::Sampler> m_samplerLinear;

    rad::Ref<vkpp::CommandStream> m_cmdStream;
    uint32_t m_cmdBufferIndex = 0;

    struct GuiPass
    {
        std::vector<rad::Ref<vkpp::CommandBuffer>> cmdBuffers;
    } m_guiPass;

    // Blend the colorImage and overlay to swapchain for present.
    struct PresentPass
    {
        rad::Ref<vkpp::ShaderStageInfo> vertStage;
        rad::Ref<vkpp::ShaderStageInfo> fragStage;
        rad::Ref<vkpp::DescriptorSetLayout> descSetLayout;
        rad::Ref<vkpp::PipelineLayout> pipelineLayout;
        rad::Ref<vkpp::Pipeline> pipeline;
        rad::Ref<vkpp::DescriptorPool> descPool;
        rad::Ref<vkpp::DescriptorSet> descSet;
        std::vector<rad::Ref<vkpp::CommandBuffer>> cmdBuffers;
    } m_presentPass;

    rad::Ref<vkpp::Fence> m_frameThrottles[MaxFrameLag];
    rad::Ref<vkpp::Semaphore> m_swapchainImageAcquiredSemaphores[MaxFrameLag];
    std::vector<rad::Ref<vkpp::Semaphore>> m_presentReady;
    std::vector<rad::Ref<vkpp::Semaphore>> m_swapchainImageOwnershipSemaphores;

}; // class VulkanFrame

} // namespace sdf
