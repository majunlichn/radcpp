#include <radcpp/GPU/VulkanCommand.h>
#include <radcpp/GPU/VulkanDevice.h>


namespace rad
{

VulkanCommandPool::VulkanCommandPool(Ref<VulkanDevice> device, VulkanQueueFamily queueFamily) :
    m_device(std::move(device)),
    m_queueFamily(queueFamily)
{
    vk::CommandPoolCreateInfo createInfo({}, m_device->GetQueueFamilyIndex(queueFamily));
    m_handle = m_device->m_handle.createCommandPool(createInfo);
}

VulkanCommandPool::~VulkanCommandPool()
{
}

vk::raii::CommandBuffers VulkanCommandPool::Allocate(vk::CommandBufferLevel level, uint32_t count)
{
    vk::CommandBufferAllocateInfo allocateInfo(m_handle, level, count);
    return vk::raii::CommandBuffers(m_device->m_handle, allocateInfo);
}

} // namespace rad
