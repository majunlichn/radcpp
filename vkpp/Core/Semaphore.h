#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Semaphore : public rad::RefCounted<Semaphore>
{
public:
    Semaphore(rad::Ref<Device> device, const vk::SemaphoreCreateInfo& createInfo);
    ~Semaphore();

    vk::Semaphore GetHandle() const { return m_wrapper; }

    rad::Ref<Device> m_device;
    vk::raii::Semaphore m_wrapper = { nullptr };

}; // class Semaphore

} // namespace vkpp
