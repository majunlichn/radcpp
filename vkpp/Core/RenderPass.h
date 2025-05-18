#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Device;

class RenderPass : public rad::RefCounted<RenderPass>
{
public:
    RenderPass(rad::Ref<Device> device, const vk::RenderPassCreateInfo& createInfo);
    ~RenderPass();

    vk::RenderPass GetHandle() const { return m_wrapper; }

    rad::Ref<Device> m_device;
    vk::raii::RenderPass m_wrapper = { nullptr };

}; // class RenderPass

} // namespace vkpp
