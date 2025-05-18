#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Device;

class Fence : public rad::RefCounted<Fence>
{
public:
    Fence(rad::Ref<Device> device, const vk::FenceCreateInfo& createInfo);
    ~Fence();

    vk::Fence GetHandle() const { return m_wrapper; }

    // @param timeout: in nanoseconds, will be adjusted to the closest value allowed by implementation.
    vk::Result Wait(uint64_t timeout = UINT64_MAX);
    void Reset();

    rad::Ref<Device> m_device;
    vk::raii::Fence m_wrapper = { nullptr };

}; // class Fence

} // namespace vkpp
