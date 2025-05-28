#include <vkpp/Core/Descriptor.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Buffer.h>
#include <vkpp/Core/Image.h>

#include <rad/Container/SmallVector.h>

namespace vkpp
{

DescriptorPool::DescriptorPool(rad::Ref<Device> device, const vk::DescriptorPoolCreateInfo& createInfo) :
    m_device(std::move(device))
{
    m_wrapper = m_device->m_wrapper.createDescriptorPool(createInfo);
}

DescriptorPool::~DescriptorPool()
{
}

std::vector<rad::Ref<DescriptorSet>> DescriptorPool::Allocate(vk::ArrayProxy<vk::DescriptorSetLayout> layouts)
{
    vk::DescriptorSetAllocateInfo allocateInfo = {};
    allocateInfo.descriptorPool = m_wrapper;
    allocateInfo.setSetLayouts(layouts);
    std::vector<vk::DescriptorSet> descSetsHandles(layouts.size());
    VK_CHECK(
        m_device->m_wrapper.getDispatcher()->vkAllocateDescriptorSets(m_device->GetHandle(),
            reinterpret_cast<const VkDescriptorSetAllocateInfo*>(&allocateInfo),
            reinterpret_cast<VkDescriptorSet*>(descSetsHandles.data()))
    );
    std::vector<rad::Ref<DescriptorSet>> descSets(descSetsHandles.size());
    for (size_t i = 0; i < descSets.size(); ++i)
    {
        descSets[i] = RAD_NEW DescriptorSet(this, descSetsHandles[i]);
    }
    return descSets;
}


DescriptorSetLayout::DescriptorSetLayout(rad::Ref<Device> device, const vk::DescriptorSetLayoutCreateInfo& createInfo) :
    m_device(std::move(device))
{
    m_wrapper = m_device->m_wrapper.createDescriptorSetLayout(createInfo);
}

DescriptorSetLayout::~DescriptorSetLayout()
{
}


DescriptorSet::DescriptorSet(rad::Ref<Device> device, vk::DescriptorPool descPoolHandle, vk::DescriptorSet descSetHandle) :
    m_device(std::move(device))
{
    m_wrapper = vk::raii::DescriptorSet(m_device->m_wrapper, descSetHandle, descPoolHandle);
}

DescriptorSet::DescriptorSet(rad::Ref<DescriptorPool> descPool, vk::DescriptorSet descSetHandle) :
    m_device(descPool->m_device),
    m_descPool(std::move(descPool))
{
    m_wrapper = vk::raii::DescriptorSet(m_device->m_wrapper, descSetHandle, m_descPool->GetHandle());
}

void DescriptorSet::Update(
    rad::ArrayRef<vk::WriteDescriptorSet> const& writes,
    rad::ArrayRef<vk::CopyDescriptorSet> const& copies)
{
    m_device->m_wrapper.updateDescriptorSets(writes, copies);
}

void DescriptorSet::UpdateBuffers(
    uint32_t binding, uint32_t arrayElement, vk::DescriptorType type,
    vk::ArrayProxy<const vk::DescriptorBufferInfo> bufferInfos)
{
    vk::WriteDescriptorSet write;
    write.dstSet = m_wrapper;
    write.dstBinding = binding;
    write.dstArrayElement = arrayElement;
    write.descriptorType = type;
    write.setBufferInfo(bufferInfos);
    m_device->m_wrapper.updateDescriptorSets(write, {});
}

void DescriptorSet::UpdateBuffers(
    uint32_t binding, uint32_t arrayElement, vk::DescriptorType type,
    vk::ArrayProxy<Buffer*> buffers)
{
    rad::SmallVector<vk::DescriptorBufferInfo, 8> bufferInfos(buffers.size());
    for (uint32_t i = 0; i < buffers.size(); i++)
    {
        bufferInfos[i].buffer = buffers.data()[i]->m_handle;
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = buffers.data()[i]->m_size;
    }

    vk::WriteDescriptorSet write = {};
    write.dstSet = m_wrapper;
    write.dstBinding = binding;
    write.dstArrayElement = arrayElement;
    write.descriptorType = type;
    write.setBufferInfo(bufferInfos);
    m_device->m_wrapper.updateDescriptorSets(write, {});
}

void DescriptorSet::UpdateCombinedImageSamplers(
    uint32_t binding, uint32_t arrayElement, rad::ArrayRef<vk::DescriptorImageInfo> imageInfos)
{
    vk::WriteDescriptorSet write = {};
    write.dstSet = this->GetHandle();
    write.dstBinding = binding;
    write.dstArrayElement = arrayElement;
    write.descriptorCount = imageInfos.size32();
    write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    write.pImageInfo = imageInfos.data();
    m_device->m_wrapper.updateDescriptorSets(write, {});
}

void DescriptorSet::UpdateCombinedImageSampler(
    uint32_t binding, uint32_t arrayElement, vk::Sampler sampler, vk::ImageView imageView, vk::ImageLayout layout)
{
    vk::DescriptorImageInfo imageInfo = {};
    imageInfo.sampler = sampler;
    imageInfo.imageView = imageView;
    imageInfo.imageLayout = layout;
    UpdateCombinedImageSamplers(binding, arrayElement, imageInfo);
}

} // namespace vkpp
