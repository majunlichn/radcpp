#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Event : public rad::RefCounted<Event>
{
public:
    Event(rad::Ref<Device> device, const vk::EventCreateInfo& createInfo);
    ~Event();

    vk::Event GetHandle() const { return m_wrapper; }

    rad::Ref<Device> m_device;
    vk::raii::Event m_wrapper = { nullptr };

}; // class Event

} // namespace vkpp
