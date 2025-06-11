#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Framebuffer : public rad::RefCounted<Framebuffer>
{
public:
    Framebuffer(rad::Ref<Device> device, const vk::FramebufferCreateInfo& createInfo);
    ~Framebuffer();

    vk::Framebuffer GetHandle() const { return m_wrapper; }

    uint32_t GetWidth() const { return m_width; }
    uint32_t GetHeight() const { return m_height; }
    uint32_t GetLayers() const { return m_layers; }

    rad::Ref<Device> m_device;
    vk::raii::Framebuffer m_wrapper = { nullptr };
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_layers = 0;

}; // class Framebuffer

} // namespace vkpp
