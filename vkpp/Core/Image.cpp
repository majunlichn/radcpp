#include <vkpp/Core/Image.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Command.h>
#include <vkpp/Core/Buffer.h>

#include <rad/IO/Image.h>

#include <vulkan/utility/vk_format_utils.h>

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

    m_cmdPool = m_device->CreateCommandPool(QueueFamily::Universal, vk::CommandPoolCreateFlagBits::eTransient);
}

Image::Image(rad::Ref<Device> device, const vk::ImageCreateInfo& imageInfo, vk::Image imageHandle) :
    m_device(std::move(device))
{
    m_handle = imageHandle;

    m_imageType = imageInfo.imageType;
    m_format = imageInfo.format;
    m_extent = imageInfo.extent;
    m_mipLevels = imageInfo.mipLevels;
    m_arrayLayers = imageInfo.arrayLayers;
    m_samples = imageInfo.samples;
    m_tiling = imageInfo.tiling;
    m_usage = imageInfo.usage;
    m_sharingMode = imageInfo.sharingMode;

    m_alloc = nullptr;  // not managed
    m_memPropFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;

    m_cmdPool = m_device->CreateCommandPool(QueueFamily::Universal, vk::CommandPoolCreateFlagBits::eTransient);
}

Image::~Image()
{
    if (m_handle && m_alloc)
    {
        vmaDestroyImage(m_device->m_allocator, m_handle, m_alloc);
    }
    m_handle = nullptr;
    m_alloc = nullptr;
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

rad::Ref<ImageView> Image::CreateView(vk::ImageViewType type)
{
    return CreateView(type, m_format);
}

rad::Ref<ImageView> Image::CreateView2D(uint32_t baseMipLevel, uint32_t levelCount, uint32_t baseArrayLayer)
{
    vk::ImageSubresourceRange range = {};
    range.aspectMask = GetImageAspectFromFormat(m_format);
    range.baseMipLevel = baseMipLevel;
    range.levelCount = levelCount;
    range.baseArrayLayer = baseArrayLayer;
    range.layerCount = 1;
    return CreateView(vk::ImageViewType::e2D, m_format, range);
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

void UploadData(Device* device, Image* image, rad::ImageU8* imageData)
{
    rad::Ref<Buffer> stagingBuffer = Buffer::CreateStagingUpload(device, imageData->m_sizeInBytes);
    stagingBuffer->WriteHost(imageData->m_data);

    rad::Ref<CommandBuffer> cmdBuffer = image->m_cmdPool->AllocatePrimary();
    cmdBuffer->Begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    vk::BufferImageCopy copy = {};
    copy.bufferOffset = 0;
    copy.bufferRowLength = imageData->m_width;
    copy.bufferImageHeight = imageData->m_height;
    copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    copy.imageSubresource.mipLevel = 0;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageOffset = vk::Offset3D{ 0, 0, 0 };
    copy.imageExtent = vk::Extent3D{ static_cast<uint32_t>(imageData->m_width), static_cast<uint32_t>(imageData->m_height), 1 };
    cmdBuffer->TransitLayoutFromCurrent(image,
        vk::PipelineStageFlagBits2::eCopy,
        vk::AccessFlagBits2::eTransferWrite,
        vk::ImageLayout::eTransferDstOptimal);
    cmdBuffer->CopyBufferToImage(stagingBuffer->GetHandle(), image->GetHandle(), image->GetCurrentLayout(), copy);
    cmdBuffer->TransitLayoutFromCurrent(image,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead,
        vk::ImageLayout::eShaderReadOnlyOptimal);
    cmdBuffer->End();
    device->GetQueue(vkpp::QueueFamily::Universal)->
        SubmitAndWaitForCompletion(cmdBuffer->GetHandle(), {}, {});
}

rad::Ref<Image> CreateTextureFromFile_R8G8B8A8_SRGB(rad::Ref<Device> device, std::string_view fileName)
{
    rad::Ref<Image> image;
    rad::Ref<rad::ImageU8> imageData = RAD_NEW rad::ImageU8();
    if (imageData->LoadFromFile(fileName, 4))
    {
        image = device->CreateImage2D_Sampled(
            vk::Format::eR8G8B8A8Srgb, imageData->m_width, imageData->m_height, 1);
        UploadData(device.get(), image.get(), imageData.get());
    }
    return image;
}

rad::Ref<Image> CreateTextureFromMemory_R8G8B8A8_SRGB(rad::Ref<Device> device, const void* buffer, size_t bufferSize)
{
    rad::Ref<Image> image;
    rad::Ref<rad::ImageU8> imageData = RAD_NEW rad::ImageU8();
    if (imageData->LoadFromMemory(buffer, bufferSize, 4))
    {
        image = device->CreateImage2D_Sampled(
            vk::Format::eR8G8B8A8Srgb, imageData->m_width, imageData->m_height, 1);
        UploadData(device.get(), image.get(), imageData.get());
    }
    return image;
}

void CopyBufferToImage(Device* device, Buffer* buffer, Image* image, rad::Span<vk::BufferImageCopy> copyInfos)
{
    rad::Ref<CommandBuffer> commandBuffer = image->m_cmdPool->AllocatePrimary();
    commandBuffer->Begin();
    // VUID-vkCmdCopyBufferToImage-dstImageLayout-01396
    if (image->GetCurrentLayout() != vk::ImageLayout::eGeneral &&
        image->GetCurrentLayout() != vk::ImageLayout::eTransferDstOptimal &&
        image->GetCurrentLayout() != vk::ImageLayout::eSharedPresentKHR)
    {
        commandBuffer->TransitLayoutFromCurrent(image,
            vk::PipelineStageFlagBits2::eCopy,
            vk::AccessFlagBits2::eTransferWrite,
            vk::ImageLayout::eTransferDstOptimal);
    }
    commandBuffer->CopyBufferToImage(buffer->GetHandle(), image->GetHandle(), image->GetCurrentLayout(), copyInfos);
    commandBuffer->TransitLayoutFromCurrent(image,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eMemoryRead,
        vk::ImageLayout::eShaderReadOnlyOptimal);
    commandBuffer->End();
    device->GetQueue(QueueFamily::Universal)->SubmitAndWaitForCompletion(commandBuffer->GetHandle(), {}, {});
}

void CopyBufferToImage2D(Device* device, Buffer* buffer, VkDeviceSize bufferOffset,
    Image* image, uint32_t baseMipLevel, uint32_t levelCount, uint32_t baseArrayLayer, uint32_t layerCount)
{
    std::vector<vk::BufferImageCopy> copyInfos(levelCount);
    vk::Extent3D blockExtent = vkuFormatTexelBlockExtent(static_cast<VkFormat>(image->GetFormat()));
    uint32_t blockSize = vkuFormatElementSize(static_cast<VkFormat>(image->GetFormat()));
    for (uint32_t i = 0; i < levelCount; i++)
    {
        uint32_t mipLevel = baseMipLevel + i;
        uint32_t mipWidth = std::max<uint32_t>(image->GetWidth() >> mipLevel, 1);
        uint32_t mipHeight = std::max<uint32_t>(image->GetHeight() >> mipLevel, 1);

        copyInfos[i].bufferOffset = bufferOffset;
        copyInfos[i].bufferRowLength = rad::RoundUpToMultiple(mipWidth, blockExtent.width);
        copyInfos[i].bufferImageHeight = rad::RoundUpToMultiple(mipHeight, blockExtent.height);
        copyInfos[i].imageSubresource.aspectMask = GetImageAspectFromFormat(image->GetFormat());
        copyInfos[i].imageSubresource.mipLevel = mipLevel;
        copyInfos[i].imageSubresource.baseArrayLayer = baseArrayLayer;
        copyInfos[i].imageSubresource.layerCount = layerCount;
        copyInfos[i].imageOffset.x = 0;
        copyInfos[i].imageOffset.y = 0;
        copyInfos[i].imageOffset.z = 0;
        copyInfos[i].imageExtent.width = mipWidth;
        copyInfos[i].imageExtent.height = mipHeight;
        copyInfos[i].imageExtent.depth = 1;

        bufferOffset += (copyInfos[i].bufferRowLength / blockExtent.width) *
            (copyInfos[i].bufferImageHeight / blockExtent.height) * blockSize * layerCount;
    }
    CopyBufferToImage(device, buffer, image, copyInfos);
}

} // namespace vkpp
