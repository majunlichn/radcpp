#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Device;
class Buffer;
class ImageView;

class Image : public rad::RefCounted<Image>
{
public:
    Image(rad::Ref<Device> device,
        const vk::ImageCreateInfo& imageInfo, const VmaAllocationCreateInfo& allocCreateInfo);
    Image(rad::Ref<Device> device,
        const vk::ImageCreateInfo& imageInfo, vk::Image imageHandle);
    ~Image();

    vk::Image GetHandle() const { return m_handle; }
    vk::Format GetFormat() const { return m_format; }
    uint32_t GetWidth() const { return m_extent.width; }
    uint32_t GetHeight() const { return m_extent.height; }
    uint32_t GetDepth() const { return m_extent.depth; }
    uint32_t GetMipLevels() const { return m_mipLevels; }
    uint32_t GetArrayLayers() const { return m_arrayLayers; }

    vk::PipelineStageFlags2 GetCurrentPipelineStage() const { return m_currentPipelineStage; }
    vk::AccessFlags2 GetCurrentAccessMask() const { return m_currentAccessFlags; }
    vk::ImageLayout GetCurrentLayout() const { return m_currentLayout; }

    void SetCurrentPipelineStage(vk::PipelineStageFlags2 stage) { m_currentPipelineStage = stage; }
    void SetCurrentAccessFlags(vk::AccessFlags2 accessFlags) { m_currentAccessFlags = accessFlags; }
    void SetCurrentLayout(vk::ImageLayout layout) { m_currentLayout = layout; }

    rad::Ref<ImageView> CreateView(
        vk::ImageViewType type, vk::Format format, const vk::ImageSubresourceRange& range,
        vk::ComponentMapping components = vk::ComponentMapping());
    rad::Ref<ImageView> CreateView(
        vk::ImageViewType type, vk::Format format);
    rad::Ref<ImageView> CreateView(vk::ImageViewType type);
    rad::Ref<ImageView> CreateView2D(
        uint32_t baseMipLevel = 0, uint32_t levelCount = 1, uint32_t baseArrayLayer = 0);

    rad::Ref<Device> m_device;
    vk::Image m_handle;
    VmaAllocation m_alloc = nullptr;
    VmaAllocationInfo m_allocInfo = {};
    vk::MemoryPropertyFlags m_memPropFlags = {};

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

    // Track image layout transition:
    vk::PipelineStageFlags2     m_currentPipelineStage = vk::PipelineStageFlagBits2::eAllCommands;
    vk::AccessFlags2            m_currentAccessFlags = vk::AccessFlagBits2::eNone;
    vk::ImageLayout             m_currentLayout = vk::ImageLayout::eUndefined;

}; // class Image

class ImageView : public rad::RefCounted<ImageView>
{
public:
    ImageView(rad::Ref<Image> image, const vk::ImageViewCreateInfo& createInfo);
    ~ImageView();

    vk::ImageView GetHandle() const { return m_wrapper; }
    Image* GetImage() const { return m_image.get(); }

    vk::Format GetFormat() const { return m_format; }

    rad::Ref<Image> m_image;
    vk::raii::ImageView m_wrapper = { nullptr };
    vk::ImageViewType m_type;
    vk::Format m_format;
    vk::ImageSubresourceRange m_range;
    vk::ComponentMapping m_components;

}; // class ImageView

rad::Ref<Image> CreateTextureFromFile_R8G8B8A8_SRGB(rad::Ref<Device> device, std::string_view fileName);
rad::Ref<Image> CreateTextureFromMemory_R8G8B8A8_SRGB(rad::Ref<Device> device, const void* buffer, size_t bufferSize);

void CopyBufferToImage(Device* device, Buffer* buffer, Image* image, rad::Span<vk::BufferImageCopy> copyInfos);
void CopyBufferToImage2D(Device* device, Buffer* buffer, VkDeviceSize bufferOffset,
    Image* image, uint32_t baseMipLevel = 0, uint32_t levelCount = 1,
    uint32_t baseArrayLayer = 0, uint32_t layerCount = 1);

} // namespace vkpp
