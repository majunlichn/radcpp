#include <vkpp/Core/Pipeline.h>
#include <vkpp/Core/Device.h>

namespace vkpp
{

GraphicsPipeline::GraphicsPipeline(
    rad::Ref<Device> device, const vk::GraphicsPipelineCreateInfo& createInfo,
    vk::Optional<const vk::raii::PipelineCache> cache) :
    Pipeline(std::move(device))
{
    m_handle = m_device->m_handle.createGraphicsPipeline(cache, createInfo, nullptr);
}

GraphicsPipeline::~GraphicsPipeline()
{
}

ComputePipeline::ComputePipeline(
    rad::Ref<Device> device, const vk::ComputePipelineCreateInfo& createInfo,
    vk::Optional<const vk::raii::PipelineCache> cache) :
    Pipeline(std::move(device))
{
    m_handle = m_device->m_handle.createComputePipeline(cache, createInfo, nullptr);
}

ComputePipeline::~ComputePipeline()
{
}

} // namespace vkpp
