#pragma once

#include <radcpp/GPU/VulkanCommon.h>

namespace rad
{

class VulkanDevice;
class VulkanBuffer;
class VulkanImage;

class VulkanDescriptorPool : public RefCounted<VulkanDescriptorPool>
{
public:
    VulkanDescriptorPool(Ref<VulkanDevice> device, const vk::DescriptorPoolCreateInfo& createInfo);
    ~VulkanDescriptorPool();

    vk::raii::DescriptorSets Allocate(vk::ArrayProxy<vk::DescriptorSetLayout> layouts);

    Ref<VulkanDevice> m_device;
    vk::raii::DescriptorPool m_handle = { nullptr };

}; // class VulkanDescriptorPool

class VulkanDescriptorUpdater
{
public:
    VulkanDescriptorUpdater(vk::raii::DescriptorSet& descSet) :
        m_descSet(descSet) {
    }
    ~VulkanDescriptorUpdater() {}

    vk::raii::DescriptorSet& m_descSet;

    void UpdateBuffers(uint32_t binding, uint32_t arrayElement, vk::DescriptorType type,
        vk::ArrayProxy<const vk::DescriptorBufferInfo> bufferInfos);
    void UpdateBuffers(uint32_t binding, uint32_t arrayElement, vk::DescriptorType type,
        vk::ArrayProxy<VulkanBuffer*> buffers);

}; // class VulkanDescriptorUpdater

} // namespace rad
