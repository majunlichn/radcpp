#include "VulkanWindow.h"
#include <SDL3/SDL_vulkan.h>

namespace sdf
{

VulkanWindow::VulkanWindow()
{
}

VulkanWindow::~VulkanWindow()
{
}

bool VulkanWindow::Create(std::string_view title, int w, int h, SDL_WindowFlags flags)
{
    if (Window::Create(title.data(), w, h, flags | SDL_WINDOW_VULKAN))
    {
        m_surface = CreateSurface();
        SDF_LOG(info, "VulkanWindow created: {} ({}x{})", title, w, h);
        return true;
    }
    else
    {
        return false;
    }
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

rad::Ref<vkpp::Swapchain> VulkanWindow::CreateSwapchain(uint32_t width, uint32_t height)
{
    assert(m_surface != nullptr);

    vk::SurfaceCapabilitiesKHR surfaceCaps = m_device->GetCapabilities(m_surface->GetHandle());
    uint32_t imageCount = 3;
    if (imageCount < surfaceCaps.minImageCount)
    {
        imageCount = surfaceCaps.minImageCount;
    }
    // If maxImageCount is 0, we can ask for as many images as we want,
    // otherwise we're limited to maxImageCount.
    if ((surfaceCaps.maxImageCount > 0) &&
        (imageCount > surfaceCaps.maxImageCount))
    {
        imageCount = surfaceCaps.maxImageCount;
    }

    vk::Format imageFormat = vk::Format::eUndefined;
    vk::ColorSpaceKHR imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;

    std::vector<vk::SurfaceFormatKHR> surfaceFormats = m_device->GetSurfaceFormats(m_surface->GetHandle());
    if (surfaceFormats[0].format == vk::Format::eUndefined)
    {
        imageFormat = vk::Format::eR8G8B8A8Unorm;
    }
    for (const auto& surfaceFormat : surfaceFormats)
    {
        const vk::Format format = surfaceFormat.format;
        if ((format == vk::Format::eR8G8B8A8Unorm) || (format == vk::Format::eB8G8R8A8Unorm) ||
            (format == vk::Format::eA2R10G10B10UnormPack32) || (format == vk::Format::eA2B10G10R10UnormPack32) ||
            (format == vk::Format::eR16G16B16A16Sfloat))
        {
            imageFormat = surfaceFormat.format;
            imageColorSpace = surfaceFormat.colorSpace;
            break;
        }
    }

    // width and height are either both 0xFFFFFFFF, or both not 0xFFFFFFFF.
    if (surfaceCaps.currentExtent.width == 0xFFFFFFFF)
    {
        // If the surface size is undefined, the size is set to the size of the images requested,
        // which must fit within the minimum and maximum values.
        if (width < surfaceCaps.minImageExtent.width)
        {
            width = surfaceCaps.minImageExtent.width;
        }
        else if (width > surfaceCaps.maxImageExtent.width)
        {
            width = surfaceCaps.maxImageExtent.width;
        }

        if (height < surfaceCaps.minImageExtent.height)
        {
            height = surfaceCaps.minImageExtent.height;
        }
        else if (height > surfaceCaps.minImageExtent.height)
        {
            height = surfaceCaps.minImageExtent.height;
        }
    }
    else
    {
        // If the surface size is defined, the swap chain size must match.
        width = surfaceCaps.minImageExtent.width;
        height = surfaceCaps.minImageExtent.height;
    }

    if ((width == 0) || (height == 0))
    {
        m_isMinimized = true;
        return m_swapchain;
    }
    else
    {
        m_isMinimized = false;
    }

    vk::SurfaceTransformFlagBitsKHR preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    if (surfaceCaps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity)
    {
        preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    }
    else
    {
        preTransform = surfaceCaps.currentTransform;
    }

    vk::CompositeAlphaFlagBitsKHR compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    std::array<vk::CompositeAlphaFlagBitsKHR, 4> compositeAlphaFlags =
    {
        vk::CompositeAlphaFlagBitsKHR::eOpaque,
        vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
        vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
        vk::CompositeAlphaFlagBitsKHR::eInherit,
    };
    for (const auto& compositeAlphaFlag : compositeAlphaFlags)
    {
        if (surfaceCaps.supportedCompositeAlpha & compositeAlphaFlag)
        {
            compositeAlpha = compositeAlphaFlag;
            break;
        }
    }

    std::vector<vk::PresentModeKHR> presentModes = m_device->GetPresentModes(m_surface->GetHandle());
    if (std::ranges::find(presentModes, m_presentMode) == std::end(presentModes))
    {
        VKPP_LOG(warn, "PresentMode {} is not supported, fallback to {}!\n",
            vk::to_string(m_presentMode),
            vk::to_string(presentModes[0]));
        m_presentMode = presentModes[0];
    }

    vk::SwapchainCreateInfoKHR swapchainInfo = {};
    swapchainInfo.flags = {};
    swapchainInfo.surface = m_surface->GetHandle();
    swapchainInfo.minImageCount = imageCount;
    swapchainInfo.imageFormat = imageFormat;
    swapchainInfo.imageColorSpace = imageColorSpace;
    swapchainInfo.imageExtent.width = width;
    swapchainInfo.imageExtent.height = height;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    swapchainInfo.imageSharingMode = vk::SharingMode::eExclusive;
    swapchainInfo.queueFamilyIndexCount = 0;
    swapchainInfo.pQueueFamilyIndices = nullptr;
    swapchainInfo.preTransform = preTransform;
    swapchainInfo.compositeAlpha = compositeAlpha;
    swapchainInfo.presentMode = m_presentMode;
    swapchainInfo.clipped = vk::True;
    swapchainInfo.oldSwapchain = m_swapchain ? m_swapchain->GetHandle() : VK_NULL_HANDLE;

    return RAD_NEW vkpp::Swapchain(m_device, swapchainInfo);
}

} // namespace sdf
