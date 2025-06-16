#pragma once

#include <vkpp/Core/Common.h>
#include <map>
#include <set>

namespace vkpp
{

class Device : public rad::RefCounted<Device>
{
public:
    Device(rad::Ref<Instance> instance, vk::raii::PhysicalDevice physicalDevice,
        const std::set<std::string>& requiredExtensions);
    ~Device();

    vk::PhysicalDevice GetPhysicalDevice() const { return m_physicalDevice; }
    vk::Device GetHandle() const { return static_cast<vk::Device>(m_wrapper); }
    const DeviceDispatcher* GetDispatcher() const { return m_wrapper.getDispatcher(); }
    PFN_vkVoidFunction GetProcAddr(const char* name) const
    {
        return m_wrapper.getProcAddr(name);
    }

    const char* GetName() const { return m_properties.deviceName; }

    rad::Ref<Instance> m_instance;
    vk::raii::PhysicalDevice m_physicalDevice;
    vk::raii::Device m_wrapper = { nullptr };
    std::array<uint32_t, size_t(QueueFamily::Count)> m_queueFamilyIndices;

    vk::SurfaceCapabilitiesKHR GetCapabilities(vk::SurfaceKHR surface) const;
    std::vector<vk::SurfaceFormatKHR> GetSurfaceFormats(vk::SurfaceKHR surface) const;
    std::vector<vk::PresentModeKHR> GetPresentModes(vk::SurfaceKHR surface) const;

    uint32_t GetQueueFamilyIndex(QueueFamily queueFamily) const
    {
        return m_queueFamilyIndices[size_t(queueFamily)];
    }

    void SetQueueFamilyIndex(QueueFamily queueFamily, uint32_t index)
    {
        m_queueFamilyIndices[size_t(queueFamily)] = index;
    }

    bool HasQueueFamily(QueueFamily queueFamily) const
    {
        return (m_queueFamilyIndices[size_t(queueFamily)] != VK_QUEUE_FAMILY_IGNORED);
    }

    const vk::QueueFamilyProperties& GetQueueFamilyProperties(QueueFamily queueFamily) const
    {
        return m_queueFamilyProperties[GetQueueFamilyIndex(queueFamily)];
    }

    Queue* GetQueue(QueueFamily queueFamily)
    {
        return m_queues[rad::ToUnderlying(queueFamily)].get();
    }

    std::set<std::string, rad::StringLess> m_enabledExtensions;
    bool IsExtensionEnabled(std::string_view name) const
    {
        return m_enabledExtensions.contains(name);
    }

    rad::Ref<CommandPool> CreateCommandPool(QueueFamily queueFamily,
        vk::CommandPoolCreateFlags flags = {});

    rad::Ref<Fence> CreateFence(vk::FenceCreateFlags flags = {});
    rad::Ref<Fence> CreateFenceSignaled();
    rad::Ref<Semaphore> CreateSemaphore(vk::SemaphoreCreateFlags flags = {});
    rad::Ref<Event> CreateEvent(vk::EventCreateFlags flags);
    void WaitIdle();

    rad::Ref<DescriptorPool> CreateDescriptorPool(
        uint32_t maxSets, rad::ArrayRef<vk::DescriptorPoolSize> poolSizes,
        vk::DescriptorPoolCreateFlags flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
    rad::Ref<DescriptorSetLayout> CreateDescriptorSetLayout(
        rad::ArrayRef<vk::DescriptorSetLayoutBinding> bindings);
    rad::Ref<PipelineLayout> CreatePipelineLayout(
        rad::ArrayRef<vk::DescriptorSetLayout> setLayouts,
        rad::ArrayRef<vk::PushConstantRange> pushConstantRanges = {});

    vk::Format FindFormat(rad::ArrayRef<vk::Format> candidates,
        vk::FormatFeatureFlags linearTilingFeatures,
        vk::FormatFeatureFlags optimalTilingFeatures,
        vk::FormatFeatureFlags bufferFeatures);

    rad::Ref<Image> CreateImage2D(
        vk::Format format, uint32_t width, uint32_t height,
        uint32_t mipLevels, vk::ImageUsageFlags usage);
    rad::Ref<Image> CreateImage2D_Sampled(vk::Format format, uint32_t width, uint32_t height, uint32_t mipLevels);
    rad::Ref<Image> CreateImage2D_ColorAttachment(vk::Format format, uint32_t width, uint32_t height);
    rad::Ref<Image> CreateImage2D_DepthStencilAttachment(vk::Format format, uint32_t width, uint32_t height);

    rad::Ref<Sampler> CreateSampler(const vk::SamplerCreateInfo& createInfo);

    rad::Ref<RenderPass> CreateRenderPass(const vk::RenderPassCreateInfo& createInfo);
    rad::Ref<Framebuffer> CreateFramebuffer(const vk::FramebufferCreateInfo& createInfo);
    rad::Ref<Framebuffer> CreateFramebuffer(vk::RenderPass renderPass, rad::ArrayRef<vk::ImageView> attachments,
        uint32_t width, uint32_t height, uint32_t layers = 1);

    rad::Ref<PipelineLayout> CreateLayout(const vk::PipelineLayoutCreateInfo& createInfo);
    rad::Ref<PipelineLayout> CreateLayout(vk::PipelineLayoutCreateFlags flags, rad::ArrayRef<vk::DescriptorSetLayout> setLayouts,
        rad::ArrayRef<vk::PushConstantRange> pushConstantRanges);

    rad::Ref<ShaderModule> CreateShaderModule(rad::ArrayRef<uint32_t> code);
    rad::Ref<GraphicsPipeline> CreateGraphicsPipeline(const vk::GraphicsPipelineCreateInfo& createInfo);
    rad::Ref<ComputePipeline> CreateComputePipeline(
        rad::Ref<ShaderStageInfo> shaderStage, vk::PipelineLayout layout);

    rad::Ref<Swapchain> CreateSwapchain(const vk::SwapchainCreateInfoKHR& createInfo);
    vk::Result Present(QueueFamily queueFamily, const vk::PresentInfoKHR& presentInfo);

    vk::PhysicalDeviceProperties m_properties;
    vk::PhysicalDeviceProperties2 m_properties2;
    vk::PhysicalDeviceVulkan11Properties m_vk11Properties;
    vk::PhysicalDeviceVulkan12Properties m_vk12Properties;
    vk::PhysicalDeviceVulkan13Properties m_vk13Properties;

    std::vector<vk::QueueFamilyProperties> m_queueFamilyProperties;
    vk::PhysicalDeviceMemoryProperties m_memoryProperties;

    vk::PhysicalDeviceFeatures m_features;
    vk::PhysicalDeviceFeatures2 m_features2;
    vk::PhysicalDeviceVulkan11Features m_Vulkan11Features;
    vk::PhysicalDeviceVulkan12Features m_Vulkan12Features;
    vk::PhysicalDeviceVulkan13Features m_Vulkan13Features;

    VmaAllocator m_allocator = nullptr;

    rad::Ref<Queue> m_queues[rad::ToUnderlying(QueueFamily::Count)];

}; // class Device

class Queue : public rad::RefCounted<Queue>
{
public:
    Queue(Device* device, uint32_t queueFamilyIndex, uint32_t queueIndex);
    ~Queue();

    vk::Queue GetHandle() const { return m_wrapper; }

    void Submit(rad::ArrayRef<vk::SubmitInfo> submitInfos, vk::Fence fence);
    void Submit(rad::ArrayRef<vk::CommandBuffer> cmdBuffers,
        rad::ArrayRef<SubmitWaitInfo> waits, rad::ArrayRef<vk::Semaphore> signalSemaphores, vk::Fence fence);
    void SubmitAndWaitForCompletion(rad::ArrayRef<vk::SubmitInfo> submitInfos);
    void SubmitAndWaitForCompletion(rad::ArrayRef<vk::CommandBuffer> cmdBuffers,
        rad::ArrayRef<SubmitWaitInfo> waits, rad::ArrayRef<vk::Semaphore> signalSemaphores);

    Device* m_device;
    uint32_t m_queueFamilyIndex;
    uint32_t m_queueIndex;
    vk::raii::Queue m_wrapper = { nullptr };

}; // class Queue

} // namespace vkpp
