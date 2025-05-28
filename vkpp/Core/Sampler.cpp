#include <vkpp/Core/Sampler.h>
#include <vkpp/Core/Device.h>

namespace vkpp
{

Sampler::Sampler(rad::Ref<Device> device, const vk::SamplerCreateInfo& createInfo) :
    m_device(std::move(device))
{
    m_wrapper = m_device->m_wrapper.createSampler(createInfo);
}

Sampler::~Sampler()
{
}

} // namespace vkpp
