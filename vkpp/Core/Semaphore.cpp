#include <vkpp/Core/Semaphore.h>
#include <vkpp/Core/Device.h>

namespace vkpp
{

Semaphore::Semaphore(rad::Ref<Device> device, const vk::SemaphoreCreateInfo& createInfo) :
    m_device(std::move(device))
{
    m_wrapper = m_device->m_wrapper.createSemaphore(createInfo);
}

Semaphore::~Semaphore()
{
}

} // namespace vkpp
