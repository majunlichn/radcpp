#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Device;

class Pipeline : public rad::RefCounted<Pipeline>
{
public:
    Pipeline(rad::Ref<Device> device) : m_device(std::move(device)) {}
    rad::Ref<Device> m_device;
    vk::raii::Pipeline m_handle = { nullptr };
}; // class Pipeline

class GraphicsPipeline : public Pipeline
{
public:
    GraphicsPipeline(rad::Ref<Device> device,
        const vk::GraphicsPipelineCreateInfo& createInfo,
        vk::Optional<const vk::raii::PipelineCache> cache);
    ~GraphicsPipeline();
}; // class GraphicsPipeline

class ComputePipeline : public Pipeline
{
public:
    ComputePipeline(rad::Ref<Device> device,
        const vk::ComputePipelineCreateInfo& createInfo,
        vk::Optional<const vk::raii::PipelineCache> cache);
    ~ComputePipeline();
}; // class ComputePipeline

} // namespace vkpp
