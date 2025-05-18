#include <vkpp/Core/Framebuffer.h>
#include <vkpp/Core/Device.h>

namespace vkpp
{

Framebuffer::Framebuffer(
    rad::Ref<Device> device, const vk::FramebufferCreateInfo& createInfo) :
    m_device(device)
{
    m_wrapper = m_device->m_wrapper.createFramebuffer(createInfo);
    m_width = createInfo.width;
    m_height = createInfo.height;
    m_layers = createInfo.layers;
}

Framebuffer::~Framebuffer()
{
}

} // namespace vkpp
