#pragma once

#include <vkpp/Core/Common.h>
#include <map>
#include <set>

namespace vkpp
{

class Instance;
class CommandPool;
class CommandBuffer;
class DescriptorPool;
class Buffer;
class BufferView;
class Image;
class ImageView;
class RenderPass;
class Framebuffer;
class ShaderStageInfo;
class Pipeline;
class GraphicsPipeline;
class ComputePipeline;

class Device : public rad::RefCounted<Device>
{
public:
    Device(rad::Ref<Instance> instance, vk::raii::PhysicalDevice physicalDevice,
        const std::set<std::string>& requiredExtensions);
    ~Device();

    vk::Device GetHandle() const { return static_cast<vk::Device>(m_wrapper); }
    const DeviceDispatcher* GetDispatcher() const { return m_wrapper.getDispatcher(); }

    const char* GetName() const { return m_properties.deviceName; }

    rad::Ref<Instance> m_instance;
    vk::raii::PhysicalDevice m_physicalDevice;
    vk::raii::Device m_wrapper = { nullptr };
    std::array<uint32_t, size_t(QueueFamily::Count)> m_queueFamilyIndices;

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

    std::set<std::string, rad::StringLess> m_enabledExtensions;
    bool IsExtensionEnabled(std::string_view name) const
    {
        return m_enabledExtensions.contains(name);
    }

    rad::Ref<CommandBuffer> AllocateTemporaryCommandBuffer(QueueFamily queueFamily);

    rad::Ref<CommandPool> CreateCommandPool(QueueFamily queueFamily, vk::CommandPoolCreateFlags flags);

    rad::Ref<DescriptorPool> CreateDescriptorPool(
        uint32_t maxSets, rad::ArrayRef<vk::DescriptorPoolSize> poolSizes,
        vk::DescriptorPoolCreateFlags flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
    vk::raii::DescriptorSetLayout CreateDescriptorSetLayout(
        rad::ArrayRef<vk::DescriptorSetLayoutBinding> bindings);
    vk::raii::PipelineLayout CreatePipelineLayout(
        rad::ArrayRef<vk::DescriptorSetLayout> setLayouts,
        rad::ArrayRef<vk::PushConstantRange> pushConstantRanges = {});

    vk::Format FindFormat(rad::ArrayRef<vk::Format> candidates,
        vk::FormatFeatureFlags linearTilingFeatures,
        vk::FormatFeatureFlags optimalTilingFeatures,
        vk::FormatFeatureFlags bufferFeatures);

    rad::Ref<Image> CreateImage2DColorAttachment(vk::Format format, uint32_t width, uint32_t height,
        vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment);
    rad::Ref<Image> CreateImage2DDepthStencilAttachment(vk::Format format, uint32_t width, uint32_t height,
        vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eDepthStencilAttachment);

    rad::Ref<RenderPass> CreateRenderPass(const vk::RenderPassCreateInfo& createInfo);
    rad::Ref<Framebuffer> CreateFramebuffer(const vk::FramebufferCreateInfo& createInfo);
    rad::Ref<Framebuffer> CreateFramebuffer(vk::RenderPass renderPass, rad::ArrayRef<vk::ImageView> attachments,
        uint32_t width, uint32_t height, uint32_t layers = 1);

    rad::Ref<ComputePipeline> CreateComputePipeline(
        rad::Ref<ShaderStageInfo> shaderStage, vk::PipelineLayout layout);

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

    vk::Queue m_queues[rad::ToUnderlying(QueueFamily::Count)];
    void Execute(rad::ArrayRef<vk::SubmitInfo> submitInfos, vk::Fence fence);
    void ExecuteSync(rad::ArrayRef<vk::SubmitInfo> submitInfos);
    void Execute(rad::ArrayRef<SubmitWaitInfo> waits, rad::ArrayRef<vk::CommandBuffer> cmdBuffers, rad::ArrayRef<vk::Semaphore> signalSemaphores, vk::Fence fence);
    void ExecuteSync(rad::ArrayRef<SubmitWaitInfo> waits, rad::ArrayRef<vk::CommandBuffer> cmdBuffers, rad::ArrayRef<vk::Semaphore> signalSemaphores);

    // Internal command pools for transient allocation.
    std::shared_ptr<vk::raii::CommandPool> m_cmdPools[rad::ToUnderlying(QueueFamily::Count)];

}; // class Device

} // namespace vkpp
