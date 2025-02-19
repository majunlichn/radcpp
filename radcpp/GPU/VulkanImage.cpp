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

} // namespace rad
