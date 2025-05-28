#pragma once

#include <vkpp/Core/Common.h>


namespace vkpp
{

class Device;

class Sampler : public rad::RefCounted<Sampler>
{
public:
    Sampler(rad::Ref<Device> device, const vk::SamplerCreateInfo& createInfo);
    ~Sampler();

    vk::Sampler GetHandle() const { return m_wrapper; }

    rad::Ref<Device> m_device;
    vk::raii::Sampler m_wrapper = { nullptr };

}; // class Sampler

} // namespace vkpp
