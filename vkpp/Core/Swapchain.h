#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Swapchain : public rad::RefCounted<Swapchain>
{
public:
    Swapchain(rad::Ref<Device> device, const vk::SwapchainCreateInfoKHR& createInfo);
    ~Swapchain();

    // Retain the surface.
    void SetSurface(rad::Ref<Surface> surface) { m_surface = std::move(surface); }

    vk::SwapchainKHR GetHandle() const { return m_wrapper; }
    uint32_t GetImageCount() const { return m_imageCount; }
    vk::Format GetFormat() const { return m_format; }
    uint32_t GetWidth() const { return m_width; }
    uint32_t GetHeight() const { return m_height; }
    vk::PresentModeKHR GetPresentMode() const { return m_presentMode; }

    Image* GetImage(uint32_t index) { return m_images[index].get(); }
    ImageView* GetDefaultImageView(uint32_t index) { return m_imageViews[index].get(); }

    // @param timeout: indicates how long the function waits, in nanoseconds, if no image is available.
    vk::Result AcquireNextImage(
        uint64_t timeout, Semaphore* semaphore, Fence* fence = nullptr, uint32_t deviceMask = 0xFFFFFFFF);

    uint32_t GetCurrentImageIndex() const { return m_currentImageIndex; }
    Image* GetCurrentImage() { return m_images[m_currentImageIndex].get(); }
    ImageView* GetCurrentImageView() { return m_imageViews[m_currentImageIndex].get(); }

    rad::Ref<Device>                    m_device;
    rad::Ref<Surface>                   m_surface;
    vk::raii::SwapchainKHR              m_wrapper = { nullptr };
    vk::Format                          m_format = vk::Format::eUndefined;
    vk::ColorSpaceKHR                   m_colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    uint32_t                            m_imageCount = 0;
    uint32_t                            m_width = 0;
    uint32_t                            m_height = 0;
    vk::PresentModeKHR                  m_presentMode = vk::PresentModeKHR::eFifo;
    std::vector<rad::Ref<Image>>        m_images;
    std::vector<rad::Ref<ImageView>>    m_imageViews;
    uint32_t                            m_currentImageIndex = 0;

}; // class Swapchain

} // namespace vkpp
