#include <vkpp/Core/Image.h>
#include <vkpp/Core/Device.h>

namespace vkpp
{

Image::Image(
    rad::Ref<Device> device, const vk::ImageCreateInfo& imageInfo,
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

Image::~Image()
{
    if (m_handle && m_alloc)
    {
        vmaDestroyImage(m_device->m_allocator, m_handle, m_alloc);
        m_handle = nullptr;
        m_alloc = nullptr;
    }
}

rad::Ref<ImageView> Image::CreateView(
    vk::ImageViewType type, vk::Format format, const vk::ImageSubresourceRange& range,
    vk::ComponentMapping components)
{
    vk::ImageViewCreateInfo viewInfo;
    viewInfo.image = m_handle;
    viewInfo.viewType = type;
    viewInfo.format = format;
    viewInfo.components = components;
    viewInfo.subresourceRange = range;
    return RAD_NEW ImageView(this, viewInfo);
}

rad::Ref<ImageView> Image::CreateView(vk::ImageViewType type, vk::Format format)
{
    vk::ImageSubresourceRange range;
    range.aspectMask = vkpp::GetDefaultImageAspectFlags(m_format);
    range.baseMipLevel = 0;
    range.levelCount = m_mipLevels;
    range.baseArrayLayer = 0;
    range.layerCount = m_arrayLayers;
    return CreateView(type, format, range);
}

rad::Ref<ImageView> Image::CreateView()
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

ImageView::ImageView(rad::Ref<Image> image, const vk::ImageViewCreateInfo& createInfo) :
    m_image(std::move(image))
{
    m_wrapper = m_image->m_device->m_wrapper.createImageView(createInfo);
    m_type = createInfo.viewType;
    m_format = createInfo.format;
    m_range = createInfo.subresourceRange;
    m_components = createInfo.components;
}

ImageView::~ImageView()
{
}

} // namespace vkpp
