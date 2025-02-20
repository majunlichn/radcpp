#include <radcpp/GPU/VulkanImage.h>
#include <radcpp/GPU/VulkanDevice.h>

namespace rad
{

VulkanImage::VulkanImage(
    Ref<VulkanDevice> device, const vk::ImageCreateInfo& imageInfo,
    const VmaAllocationCreateInfo& allocCreateInfo) :
    m_device(std::move(device))
{
    VK_CHECK(vmaCreateImage(m_device->m_allocator,
        reinterpret_cast<const VkImageCreateInfo*>(&imageInfo),
        &allocCreateInfo,
        reinterpret_cast<VkImage*>(&m_handle),
        &m_alloc,
        &m_allocInfo));
    if (m_handle && m_alloc)
    {
        m_flags = imageInfo.flags;
        m_imageType = imageInfo.imageType;
        m_format = imageInfo.format;
        m_extent = imageInfo.extent;
        m_mipLevels = imageInfo.mipLevels;
        m_arrayLayers = imageInfo.arrayLayers;
        m_samples = imageInfo.samples;
        m_tiling = imageInfo.tiling;
        m_usage = imageInfo.usage;
        m_sharingMode = imageInfo.sharingMode;
        vmaGetAllocationMemoryProperties(m_device->m_allocator, m_alloc,
            reinterpret_cast<VkMemoryPropertyFlags*>(&m_memPropFlags));


    }
}

VulkanImage::~VulkanImage()
{
    if (m_handle && m_alloc)
    {
        vmaDestroyImage(m_device->m_allocator, m_handle, m_alloc);
        m_handle = nullptr;
        m_alloc = nullptr;
    }
}

Ref<VulkanImageView> VulkanImage::CreateView(
    vk::ImageViewType type, vk::Format format, const vk::ImageSubresourceRange& range,
    vk::ComponentMapping components)
{
    vk::ImageViewCreateInfo viewInfo;
    viewInfo.image = m_handle;
    viewInfo.viewType = type;
    viewInfo.format = format;
    viewInfo.components = components;
    viewInfo.subresourceRange = range;
    return RAD_NEW VulkanImageView(this, viewInfo);
}

Ref<VulkanImageView> VulkanImage::CreateView(vk::ImageViewType type, vk::Format format)
{
    vk::ImageSubresourceRange range;
    range.aspectMask = vkpp::GetDefaultImageAspectFlags(m_format);
    range.baseMipLevel = 0;
    range.levelCount = m_mipLevels;
    range.baseArrayLayer = 0;
    range.layerCount = m_arrayLayers;
    return CreateView(type, format, range);
}

Ref<VulkanImageView> VulkanImage::CreateView()
{
    vk::ImageViewType viewType;
    if (m_imageType == vk::ImageType::e1D)
    {
        viewType = vk::ImageViewType::e1D;
    }
    else if (m_imageType == vk::ImageType::e2D)
    {
        viewType = vk::ImageViewType::e2D;
    }
    else if (m_imageType == vk::ImageType::e3D)
    {
        viewType = vk::ImageViewType::e3D;
    }
    return CreateView(viewType, m_format);
}

VulkanImageView::VulkanImageView(Ref<VulkanImage> image, const vk::ImageViewCreateInfo& createInfo) :
    m_image(std::move(image))
{
    m_handle = m_image->m_device->m_handle.createImageView(createInfo);
    m_type = createInfo.viewType;
    m_format = createInfo.format;
    m_range = createInfo.subresourceRange;
    m_components = createInfo.components;
}

VulkanImageView::~VulkanImageView()
{
}

} // namespace rad
