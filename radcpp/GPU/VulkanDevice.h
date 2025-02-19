#pragma once

#include <radcpp/GPU/VulkanCommon.h>
#include <map>
#include <set>

namespace rad
{

class VulkanInstance;

class VulkanDevice : public RefCounted<VulkanDevice>
{
public:
    VulkanDevice(Ref<VulkanInstance> instance, vk::raii::PhysicalDevice physicalDevice,
        const std::set<std::string>& requiredExtensions);
    ~VulkanDevice();

    vk::Device GetHandle() const { return static_cast<vk::Device>(m_handle); }
    const char* GetName() const { return m_properties.deviceName; }

    Ref<VulkanInstance> m_instance;
    vk::raii::PhysicalDevice m_physicalDevice;
    vk::raii::Device m_handle = { nullptr };
    std::array<uint32_t, size_t(VulkanQueueFamily::Count)> m_queueFamilyIndices;
    uint32_t GetQueueFamilyIndex(VulkanQueueFamily queueFamily)
    {
        return m_queueFamilyIndices[size_t(queueFamily)];
    }
    void SetQueueFamilyIndex(VulkanQueueFamily queueFamily, uint32_t index)
    {
        m_queueFamilyIndices[size_t(queueFamily)] = index;
    }
    bool HasQueueFamily(VulkanQueueFamily queueFamily) const
    {
        return (m_queueFamilyIndices[size_t(queueFamily)] != VK_QUEUE_FAMILY_IGNORED);
    }

    std::set<std::string, StringLess> m_enabledExtensions;
    bool IsExtensionEnabled(std::string_view name) const
    {
        return m_enabledExtensions.contains(name);
    }

    vk::Format FindFormat(Span<const vk::Format> candidates,
        vk::FormatFeatureFlags linearTilingFeatures,
        vk::FormatFeatureFlags optimalTilingFeatures,
        vk::FormatFeatureFlags bufferFeatures);

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

    vk::Queue m_queues[size_t(VulkanQueueFamily::Count)];

}; // class VulkanDevice

} // namespace rad
