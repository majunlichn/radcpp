#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Device;
class Buffer;
class Image;

class DescriptorPool : public rad::RefCounted<DescriptorPool>
{
public:
    DescriptorPool(rad::Ref<Device> device, const vk::DescriptorPoolCreateInfo& createInfo);
    ~DescriptorPool();

    vk::raii::DescriptorSets Allocate(vk::ArrayProxy<vk::DescriptorSetLayout> layouts);

    rad::Ref<Device> m_device;
    vk::raii::DescriptorPool m_handle = { nullptr };

}; // class DescriptorPool

class DescriptorUpdater
{
public:
    DescriptorUpdater(vk::raii::DescriptorSet& descSet) :
        m_descSet(descSet) {
    }
    ~DescriptorUpdater() {}

    vk::raii::DescriptorSet& m_descSet;

    void UpdateBuffers(uint32_t binding, uint32_t arrayElement, vk::DescriptorType type,
        vk::ArrayProxy<const vk::DescriptorBufferInfo> bufferInfos);
    void UpdateBuffers(uint32_t binding, uint32_t arrayElement, vk::DescriptorType type,
        vk::ArrayProxy<Buffer*> buffers);

}; // class DescriptorUpdater

} // namespace vkpp
