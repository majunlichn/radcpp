#include <vkpp/Core/Fence.h>
#include <vkpp/Core/Device.h>

namespace vkpp
{

Fence::Fence(rad::Ref<Device> device, const vk::FenceCreateInfo& createInfo) :
    m_device(std::move(device))
{
    m_wrapper = m_device->m_wrapper.createFence(createInfo);
}

Fence::~Fence()
{
}

void Fence::Wait(uint64_t timeout)
{
    m_device->m_wrapper.waitForFences({ m_wrapper }, vk::True, timeout);
}

void Fence::Reset()
{
    m_device->m_wrapper.resetFences({ m_wrapper });
}

} // namespace vkpp
