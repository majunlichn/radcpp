#pragma once

#include <SDFramework/Core/Common.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Command.h>
#include <vkpp/Core/Fence.h>
#include <vkpp/Core/Semaphore.h>
#include <vkpp/Core/Buffer.h>
#include <vkpp/Core/Image.h>
#include <vkpp/Core/Descriptor.h>
#include <vkpp/Core/Pipeline.h>

#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>

namespace sdf
{

class VulkanWindow;

class VulkanGuiContext : public rad::RefCounted<VulkanGuiContext>
{
public:
    VulkanGuiContext(VulkanWindow* window, rad::Ref<vkpp::Device> device);
    ~VulkanGuiContext();

    bool Init();
    void Destroy();
    bool ProcessEvent(const SDL_Event& event);

    void NewFrame();
    void Render();

    VulkanWindow* m_window;
    rad::Ref<vkpp::Device> m_device;
    ImGuiContext* m_context = nullptr;

    rad::Ref<vkpp::Image> m_renderTarget;
    rad::Ref<vkpp::ImageView> m_renderTargetView;
    rad::Ref<vkpp::Image> m_overlay;
    rad::Ref<vkpp::ImageView> m_overlayView;

    static constexpr size_t FrameLag = 2;

    rad::Ref<vkpp::Buffer> m_uniformBuffers[FrameLag];
    void* m_uniforms[FrameLag] = {};

    rad::Ref<vkpp::DescriptorPool> m_descPool;
    rad::Ref<vkpp::DescriptorSet> m_descSets[FrameLag];

    rad::Ref<vkpp::CommandPool> m_cmdPool;
    rad::Ref<vkpp::CommandPool> m_presentCmdPool;
    rad::Ref<vkpp::CommandBuffer> m_cmdBuffers[FrameLag];
    rad::Ref<vkpp::CommandBuffer> m_cmdBuffers_GraphicsToPresent[FrameLag];
    uint32_t m_cmdBufferIndex = 0;
    rad::Ref<vkpp::Fence> m_fences[FrameLag];
    rad::Ref<vkpp::Semaphore> m_imageAcquiredSemaphores[FrameLag];

    std::vector<rad::Ref<vkpp::Semaphore>> m_drawCompletedSemaphores;
    std::vector<rad::Ref<vkpp::Semaphore>> m_swapchainImageOwnershipSemaphores;

}; // class VulkanGuiContext

} // namespace sdf
