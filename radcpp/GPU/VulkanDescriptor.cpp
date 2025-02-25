#include <radcpp/GPU/VulkanDescriptor.h>
#include <radcpp/GPU/VulkanDevice.h>
#include <radcpp/GPU/VulkanBuffer.h>
#include <radcpp/GPU/VulkanImage.h>

#include <radcpp/Container/SmallVector.h>

namespace rad
{

VulkanDescriptorPool::VulkanDescriptorPool(Ref<VulkanDevice> device, const vk::DescriptorPoolCreateInfo& createInfo) :
    m_device(std::move(device))
{
    m_handle = m_device->m_handle.createDescriptorPool(createInfo);
}

VulkanDescriptorPool::~VulkanDescriptorPool()
{
}

vk::raii::DescriptorSets VulkanDescriptorPool::Allocate(vk::ArrayProxy<vk::DescriptorSetLayout> layouts)
{
    vk::DescriptorSetAllocateInfo allocInfo = {};
    allocInfo.descriptorPool = m_handle;
    allocInfo.setSetLayouts(layouts);
    return vk::raii::DescriptorSets(m_device->m_handle, allocInfo);
}

void VulkanDescriptorUpdater::UpdateBuffers(
    uint32_t binding, uint32_t arrayElement, vk::DescriptorType type,
    vk::ArrayProxy<const vk::DescriptorBufferInfo> bufferInfos)
{
    vk::WriteDescriptorSet write;
    write.dstSet = m_descSet;
    write.dstBinding = binding;
    write.dstArrayElement = arrayElement;
    write.descriptorType = type;
    write.setBufferInfo(bufferInfos);
    m_descSet.getDevice().updateDescriptorSets(write, {}, *m_descSet.getDispatcher());
}

void VulkanDescriptorUpdater::UpdateBuffers(
    uint32_t binding, uint32_t arrayElement, vk::DescriptorType type,
    vk::ArrayProxy<VulkanBuffer*> buffers)
{
    SmallVector<vk::DescriptorBufferInfo, 8> bufferInfos(buffers.size());
    for (uint32_t i = 0; i < buffers.size(); i++)
    {
        bufferInfos[i].buffer = buffers.data()[i]->m_handle;
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = buffers.data()[i]->m_size;
    }

    vk::WriteDescriptorSet write = {};
    write.dstSet = m_descSet;
    write.dstBinding = binding;
    write.dstArrayElement = arrayElement;
    write.descriptorType = type;
    write.setBufferInfo(bufferInfos);
    m_descSet.getDevice().updateDescriptorSets(write, {}, *m_descSet.getDispatcher());
}

} // namespace rad
