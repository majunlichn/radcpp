#include <vkpp/Core/Event.h>
#include <vkpp/Core/Device.h>

namespace vkpp
{

Event::Event(rad::Ref<Device> device, const vk::EventCreateInfo& createInfo) :
    m_device(std::move(device))
{
    m_wrapper = m_device->m_wrapper.createEvent(createInfo);
}

Event::~Event()
{
}

} // namespace vkpp
