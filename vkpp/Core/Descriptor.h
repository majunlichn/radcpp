#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class DescriptorPool : public rad::RefCounted<DescriptorPool>
{
public:
    DescriptorPool(rad::Ref<Device> device, const vk::DescriptorPoolCreateInfo& createInfo);
    ~DescriptorPool();

    vk::DescriptorPool GetHandle() const { return m_wrapper; }

    std::vector<rad::Ref<DescriptorSet>> Allocate(vk::ArrayProxy<vk::DescriptorSetLayout> layouts);

    rad::Ref<Device> m_device;
    vk::raii::DescriptorPool m_wrapper = { nullptr };

}; // class DescriptorPool

class DescriptorSetLayout : public rad::RefCounted<DescriptorSetLayout>
{
public:
    DescriptorSetLayout(rad::Ref<Device> device, const vk::DescriptorSetLayoutCreateInfo& createInfo);
    ~DescriptorSetLayout();

    vk::DescriptorSetLayout GetHandle() const { return m_wrapper; }

    rad::Ref<Device> m_device;
    vk::raii::DescriptorSetLayout m_wrapper = { nullptr };

}; // class DescriptorSetLayout

class DescriptorSet : public rad::RefCounted<DescriptorSet>
{
public:
    DescriptorSet(rad::Ref<DescriptorPool> descPool, vk::DescriptorSet descSetHandle);
    ~DescriptorSet() {}

    vk::DescriptorSet GetHandle() const { return m_wrapper; }

    rad::Ref<DescriptorPool> m_descPool;
    vk::raii::DescriptorSet m_wrapper = { nullptr };

    void Update(
        rad::ArrayRef<vk::WriteDescriptorSet> const& writes,
        rad::ArrayRef<vk::CopyDescriptorSet> const& copies);

    void UpdateBuffers(uint32_t binding, uint32_t arrayElement, vk::DescriptorType type,
        vk::ArrayProxy<const vk::DescriptorBufferInfo> bufferInfos);
    void UpdateBuffers(uint32_t binding, uint32_t arrayElement, vk::DescriptorType type,
        vk::ArrayProxy<Buffer*> buffers);

    void UpdateCombinedImageSamplers(uint32_t binding, uint32_t arrayElement,
        rad::ArrayRef<vk::DescriptorImageInfo> imageInfos);
    void UpdateCombinedImageSampler(uint32_t binding, uint32_t arrayElement,
        vk::Sampler sampler, vk::ImageView imageView, vk::ImageLayout layout);

}; // class DescriptorSet

} // namespace vkpp
