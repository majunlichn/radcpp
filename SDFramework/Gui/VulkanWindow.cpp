#include "VulkanWindow.h"
#include <SDL3/SDL_vulkan.h>

namespace sdf
{

VulkanWindow::VulkanWindow(rad::Ref<vkpp::Instance> instance) :
    m_instance(instance)
{
}

VulkanWindow::~VulkanWindow()
{
}

bool VulkanWindow::Create(const char* title, int w, int h, SDL_WindowFlags flags)
{
    if (!Window::Create(title, w, h, flags | SDL_WINDOW_VULKAN))
    {
        return false;
    }
    m_surface = CreateSurface();
    SDF_LOG(info, "VulkanWindow created: {} ({}x{})", title, w, h);
    return true;
}

void VulkanWindow::Destroy()
{
}

rad::Ref<vkpp::Surface> VulkanWindow::CreateSurface()
{
    VkSurfaceKHR surfaceHandle = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(m_handle, m_instance->GetHandle(), nullptr, &surfaceHandle))
    {
        SDF_LOG(err, "SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
    }
    return RAD_NEW vkpp::Surface(m_instance, surfaceHandle);
}

rad::Ref<vkpp::Swapchain> VulkanWindow::CreateSwapchain(int width, int height)
{
    // width and height are either both 0xFFFFFFFF, or both not 0xFFFFFFFF.
    if (m_surfaceCaps.currentExtent.width == 0xFFFFFFFF)
    {
        // If the surface size is undefined, the size is set to the size of the images requested,
        // which must fit within the minimum and maximum values.
        if (width < m_surfaceCaps.minImageExtent.width)
        {
            width = m_surfaceCaps.minImageExtent.width;
        }
        else if (width > m_surfaceCaps.maxImageExtent.width)
        {
            width = m_surfaceCaps.maxImageExtent.width;
        }

        if (height < m_surfaceCaps.minImageExtent.height)
        {
            height = m_surfaceCaps.minImageExtent.height;
        }
        else if (height > m_surfaceCaps.minImageExtent.height)
        {
            height = m_surfaceCaps.minImageExtent.height;
        }
    }
    else
    {
        // If the surface size is defined, the swap chain size must match
        width = m_surfaceCaps.minImageExtent.width;
        height = m_surfaceCaps.minImageExtent.height;
    }

    uint32_t imageCount = 3;
    if (imageCount < m_surfaceCaps.minImageCount)
    {
        imageCount = m_surfaceCaps.minImageCount;
    }
    if ((m_surfaceCaps.maxImageCount > 0) &&
        (imageCount > m_surfaceCaps.maxImageCount))
    {
        imageCount = m_surfaceCaps.maxImageCount;
    }

    m_surfaceFormat = m_surfaceFormats[0];
    if (m_surfaceFormat.format == vk::Format::eUndefined)
    {
        m_surfaceFormat.format = vk::Format::eR8G8B8A8Unorm;
    }

    for (size_t i = 0; i < m_surfaceFormats.size(); i++)
    {
        const vk::Format format = m_surfaceFormats[i].format;
        if ((format == vk::Format::eA2R10G10B10UnormPack32) ||
            (format == vk::Format::eA2B10G10R10UnormPack32) ||
            (format == vk::Format::eR16G16B16A16Sfloat))
        {
            m_surfaceFormat = m_surfaceFormats[i];
            break;
        }
    }

    VKPP_LOG(info, "Select SurfaceFormat {} ({})",
        vk::to_string(m_surfaceFormat.format),
        vk::to_string(m_surfaceFormat.colorSpace));

    vk::ColorSpaceKHR colorSpace = m_surfaceFormats[0].colorSpace;

    vk::SurfaceTransformFlagBitsKHR preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    if (m_surfaceCaps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity)
    {
        preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    }
    else
    {
        preTransform = m_surfaceCaps.currentTransform;
    }

    vk::CompositeAlphaFlagBitsKHR compositeAlpha =
        (m_surfaceCaps.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eInherit) ?
        vk::CompositeAlphaFlagBitsKHR::eInherit : vk::CompositeAlphaFlagBitsKHR::eOpaque;
    if (m_surfaceCaps.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePreMultiplied)
    {
        compositeAlpha = vk::CompositeAlphaFlagBitsKHR::ePreMultiplied;
    }
    else if (m_surfaceCaps.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePostMultiplied)
    {
        compositeAlpha = vk::CompositeAlphaFlagBitsKHR::ePostMultiplied;
    }

    if (std::ranges::find(m_presentModes, m_presentMode) == std::end(m_presentModes))
    {
        VKPP_LOG(warn, "PresentMode {} is not supported, fallback to {}!\n",
            vk::to_string(m_presentMode),
            vk::to_string(m_presentModes[0]));
        m_presentMode = m_presentModes[0];
    }

    vk::SwapchainCreateInfoKHR swapchainInfo = {};
    swapchainInfo.flags = {};
    swapchainInfo.surface = m_surface->GetHandle();
    swapchainInfo.minImageCount = imageCount;
    swapchainInfo.imageFormat = m_surfaceFormat.format;
    swapchainInfo.imageColorSpace = colorSpace;
    swapchainInfo.imageExtent.width = width;
    swapchainInfo.imageExtent.height = height;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageUsage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment;
    swapchainInfo.imageSharingMode = vk::SharingMode::eExclusive;
    swapchainInfo.queueFamilyIndexCount = 0;
    swapchainInfo.pQueueFamilyIndices = nullptr;
    swapchainInfo.preTransform = preTransform;
    swapchainInfo.compositeAlpha = compositeAlpha;
    swapchainInfo.presentMode = m_presentMode;
    swapchainInfo.clipped = VK_TRUE;
    swapchainInfo.oldSwapchain = m_swapchain ? m_swapchain->GetHandle() : VK_NULL_HANDLE;

    return RAD_NEW vkpp::Swapchain(m_device, swapchainInfo);
}

} // namespace sdf
