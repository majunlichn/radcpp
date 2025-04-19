#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Device;
class ImageView;

class Image : public rad::RefCounted<Image>
{
public:
    Image(rad::Ref<Device> device,
        const vk::ImageCreateInfo& imageInfo, const VmaAllocationCreateInfo& allocCreateInfo);
    ~Image();

    rad::Ref<ImageView> CreateView(
        vk::ImageViewType type, vk::Format format, const vk::ImageSubresourceRange& range,
        vk::ComponentMapping components = vk::ComponentMapping());
    rad::Ref<ImageView> CreateView(
        vk::ImageViewType type, vk::Format format);
    rad::Ref<ImageView> CreateView();

    rad::Ref<Device> m_device;
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

}; // class Image

class ImageView : public rad::RefCounted<ImageView>
{
public:
    ImageView(rad::Ref<Image> image, const vk::ImageViewCreateInfo& createInfo);
    ~ImageView();

    rad::Ref<Image> m_image;
    vk::raii::ImageView m_handle = { nullptr };
    vk::ImageViewType m_type;
    vk::Format m_format;
    vk::ImageSubresourceRange m_range;
    vk::ComponentMapping m_components;

}; // class ImageView

} // namespace vkpp
