#pragma once

#include <radcpp/GPU/VulkanCommon.h>

namespace rad
{

class VulkanDevice;

class VulkanCommandPool : public RefCounted<VulkanCommandPool>
{
public:
    VulkanCommandPool(Ref<VulkanDevice> device, VulkanQueueFamily queueFamily);
    ~VulkanCommandPool();

    VkCommandPool GetHandle() const { return static_cast<vk::CommandPool>(m_handle); }

    vk::raii::CommandBuffers Allocate(vk::CommandBufferLevel level, uint32_t count);

    Ref<VulkanDevice> m_device;
    VulkanQueueFamily m_queueFamily;
    vk::raii::CommandPool m_handle = { nullptr };

}; // class VulkanCommandPool

} // namespace rad
