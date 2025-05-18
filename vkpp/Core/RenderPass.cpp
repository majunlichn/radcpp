#include <vkpp/Core/RenderPass.h>
#include <vkpp/Core/Device.h>

namespace vkpp
{

RenderPass::RenderPass(
    rad::Ref<Device> device, const vk::RenderPassCreateInfo& createInfo) :
    m_device(device)
{
    m_wrapper = m_device->m_wrapper.createRenderPass(createInfo);
}

RenderPass::~RenderPass()
{
}

} // namespace vkpp
