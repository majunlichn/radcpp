#pragma once

#include <radcpp/GPU/VulkanCommon.h>

namespace rad
{

class VulkanDevice;

class VulkanImage : public RefCounted<VulkanImage>
{
public:
    VulkanImage(Ref<VulkanDevice> device,
        const vk::ImageCreateInfo& imageInfo, const VmaAllocationCreateInfo& allocCreateInfo);
    ~VulkanImage();

    Ref<VulkanDevice> m_device;
    vk::Image m_handle;
    VmaAllocation m_alloc = nullptr;
    VmaAllocationInfo m_allocInfo = {};

    vk::ImageCreateFlags        m_flags = {};
    vk::ImageType               m_imageType = vk::ImageType::e1D;
    vk::Format                  m_format = vk::Format::eUndefined;
    vk::Extent3D                m_extent = {};
    uint32_t                    m_mipLevels = {};
    uint32_t                    m_arrayLayers = {};
    vk::SampleCountFlagBits     m_samples = vk::SampleCountFlagBits::e1;
    vk::ImageTiling             m_tiling = vk::ImageTiling::eOptimal;
    vk::ImageUsageFlags         m_usage = {};
    vk::SharingMode             m_sharingMode = vk::SharingMode::eExclusive;

    vk::MemoryPropertyFlags     m_memPropFlags;

}; // class VulkanImage

} // namespace rad
