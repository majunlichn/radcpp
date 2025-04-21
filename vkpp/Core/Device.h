#pragma once

#include <vkpp/Core/Common.h>
#include <map>
#include <set>

namespace vkpp
{

class Instance;
class CommandPool;
class DescriptorPool;
class Buffer;
class BufferView;
class Image;
class ImageView;

class Device : public rad::RefCounted<Device>
{
public:
    Device(rad::Ref<Instance> instance, vk::raii::PhysicalDevice physicalDevice,
        const std::set<std::string>& requiredExtensions);
    ~Device();

    vk::Device GetHandle() const { return static_cast<vk::Device>(m_handle); }
    const char* GetName() const { return m_properties.deviceName; }

    rad::Ref<Instance> m_instance;
    vk::raii::PhysicalDevice m_physicalDevice;
    vk::raii::Device m_handle = { nullptr };
    std::array<uint32_t, size_t(QueueFamily::Count)> m_queueFamilyIndices;
    uint32_t GetQueueFamilyIndex(QueueFamily queueFamily)
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

    std::set<std::string, rad::StringLess> m_enabledExtensions;
    bool IsExtensionEnabled(std::string_view name) const
    {
        return m_enabledExtensions.contains(name);
    }

    rad::Ref<CommandPool> CreateCommandPool(QueueFamily queueFamily, vk::CommandPoolCreateFlags flags);
    rad::Ref<DescriptorPool> CreateDescriptorPool(
        uint32_t maxSets, rad::ArrayRef<vk::DescriptorPoolSize> poolSizes,
        vk::DescriptorPoolCreateFlags flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
    vk::raii::DescriptorSetLayout CreateDescriptorSetLayout(
        rad::ArrayRef<vk::DescriptorSetLayoutBinding> bindings);


    vk::Format FindFormat(rad::ArrayRef<vk::Format> candidates,
        vk::FormatFeatureFlags linearTilingFeatures,
        vk::FormatFeatureFlags optimalTilingFeatures,
        vk::FormatFeatureFlags bufferFeatures);

    rad::Ref<Image> CreateImage2DColorAttachment(vk::Format format, uint32_t width, uint32_t height,
        vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment);
    rad::Ref<Image> CreateImage2DDepthStencilAttachment(vk::Format format, uint32_t width, uint32_t height,
        vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eDepthStencilAttachment);

    vk::PhysicalDeviceProperties m_properties;
    vk::PhysicalDeviceProperties2 m_properties2;
    vk::PhysicalDeviceVulkan11Properties m_vk11Properties;
    vk::PhysicalDeviceVulkan12Properties m_vk12Properties;
    vk::PhysicalDeviceVulkan13Properties m_vk13Properties;

    std::vector<vk::QueueFamilyProperties> m_queueFamilies;
    vk::PhysicalDeviceMemoryProperties m_memoryProperties;

    vk::PhysicalDeviceFeatures m_features;
    vk::PhysicalDeviceFeatures2 m_features2;
    vk::PhysicalDeviceVulkan11Features m_Vulkan11Features;
    vk::PhysicalDeviceVulkan12Features m_Vulkan12Features;
    vk::PhysicalDeviceVulkan13Features m_Vulkan13Features;

    VmaAllocator m_allocator = nullptr;

    vk::Queue m_queues[size_t(QueueFamily::Count)];

}; // class Device

} // namespace vkpp
