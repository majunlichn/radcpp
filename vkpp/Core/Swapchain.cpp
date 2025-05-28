#include <vkpp/Core/Swapchain.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Fence.h>
#include <vkpp/Core/Semaphore.h>
#include <vkpp/Core/Image.h>
#include <vkpp/Core/Surface.h>

namespace vkpp
{

Swapchain::Swapchain(rad::Ref<Device> device, const vk::SwapchainCreateInfoKHR& createInfo) :
    m_device(std::move(device))
{
    m_wrapper = m_device->m_wrapper.createSwapchainKHR(createInfo);

    m_imageCount = createInfo.minImageCount;
    m_format = createInfo.imageFormat;
    m_colorSpace = createInfo.imageColorSpace;
    m_width = createInfo.imageExtent.width;
    m_height = createInfo.imageExtent.height;
    m_presentMode = createInfo.presentMode;

    std::vector<vk::Image> imageHandles = m_wrapper.getImages();
    m_imageCount = static_cast<uint32_t>(imageHandles.size());

    m_images.resize(m_imageCount);
    m_imageViews.resize(m_imageCount);
    for (uint32_t i = 0; i < m_imageCount; ++i)
    {
        vk::ImageCreateInfo imageCreateInfo = {};
        imageCreateInfo.flags = {};
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.format = createInfo.imageFormat;
        imageCreateInfo.extent.width = createInfo.imageExtent.width;
        imageCreateInfo.extent.height = createInfo.imageExtent.height;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = createInfo.imageArrayLayers;
        imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
        imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
        imageCreateInfo.usage = createInfo.imageUsage;
        imageCreateInfo.sharingMode = createInfo.imageSharingMode;
        imageCreateInfo.queueFamilyIndexCount = createInfo.queueFamilyIndexCount;
        imageCreateInfo.pQueueFamilyIndices = createInfo.pQueueFamilyIndices;
        imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
        m_images[i] = RAD_NEW Image(m_device, imageCreateInfo, imageHandles[i]);
        m_imageViews[i] = m_images[i]->CreateView2D();
    }
}

Swapchain::~Swapchain()
{
}

vk::Result Swapchain::AcquireNextImage(uint64_t timeout, Semaphore* semaphore, Fence* fence, uint32_t deviceMask)
{
    vk::AcquireNextImageInfoKHR acquireInfo;
    acquireInfo.swapchain = m_wrapper;
    acquireInfo.timeout = timeout;
    acquireInfo.semaphore = semaphore ? semaphore->GetHandle() : nullptr;
    acquireInfo.fence = fence ? fence->GetHandle() : nullptr;
    acquireInfo.deviceMask = deviceMask;
    auto [result, index] = m_device->m_wrapper.acquireNextImage2KHR(acquireInfo);
    if ((result == vk::Result::eSuccess) ||
        (result == vk::Result::eSuboptimalKHR))
    {
        m_currentImageIndex = index;
    }
    return result;
}

} // namesapce vkpp
