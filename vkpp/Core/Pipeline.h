#pragma once

#include <vkpp/Core/Common.h>
#include <vkpp/Core/ShaderCompiler.h>
#include <map>

namespace vkpp
{

class Device;

class ShaderStageInfo : public rad::RefCounted<ShaderStageInfo>
{
public:
    ShaderStageInfo() {}
    ~ShaderStageInfo() {}

    static rad::Ref<ShaderStageInfo> CreateFromGLSL(rad::Ref<Device> device,
        vk::ShaderStageFlagBits stage, const std::string& fileName, const std::string& source,
        const std::string& entryPoint = "main", rad::Span<ShaderMacro> macros = {});

    template<typename T>
    void AddSpecData(uint32_t constantID, const T& value)
    {
        size_t offset = m_specData.size();
            m_specData.resize(m_specData.size() + sizeof(value));
        std::memcpy(m_specData.data() + offset, &value, sizeof(value));
        m_specMapEntries.emplace_back(constantID, static_cast<uint32_t>(offset), sizeof(T));

        m_specInfo.mapEntryCount = static_cast<uint32_t>(m_specMapEntries.size());
        m_specInfo.pMapEntries = m_specMapEntries.data();
        m_specInfo.dataSize = m_specData.size();
        m_specInfo.pData = m_specData.data();
    }

    operator vk::PipelineShaderStageCreateInfo() const
    {
        vk::PipelineShaderStageCreateInfo createInfo = {};
        createInfo.flags = m_flags;
        createInfo.stage = m_stage;
        createInfo.module = m_module;
        createInfo.pName = m_entryPoint.c_str();
        createInfo.pSpecializationInfo = nullptr;
        if (!m_specMapEntries.empty() && !m_specData.empty())
        {
            createInfo.pSpecializationInfo = &m_specInfo;
        }
        return createInfo;
    }

    vk::PipelineShaderStageCreateFlags m_flags = {};
    vk::ShaderStageFlagBits m_stage = vk::ShaderStageFlagBits::eCompute;
    vk::raii::ShaderModule m_module = { nullptr };
    std::string m_entryPoint;
    vk::SpecializationInfo m_specInfo;
    std::vector<vk::SpecializationMapEntry> m_specMapEntries;
    std::vector<uint8_t> m_specData;

}; // class ShaderStageInfo

class Pipeline : public rad::RefCounted<Pipeline>
{
public:
    Pipeline(rad::Ref<Device> device, vk::PipelineBindPoint bindPoint) :
        m_device(std::move(device)), m_bindPoint(bindPoint){}

    vk::Pipeline GetHandle() const { return m_wrapper; };
    vk::PipelineBindPoint GetBindPoint() const { return m_bindPoint; };

    void CreateLayout(const vk::PipelineLayoutCreateInfo& createInfo);
    void CreateLayout(vk::PipelineLayoutCreateFlags flags, rad::ArrayRef<vk::DescriptorSetLayout> setLayouts,
        rad::ArrayRef<vk::PushConstantRange> pushConstantRanges);

    rad::Ref<ShaderStageInfo> CreateShaderStageFromGLSL(
        vk::ShaderStageFlagBits stage, const std::string& fileName, const std::string& source,
        const std::string& entryPoint = "main", rad::Span<ShaderMacro> macros = {},
        shaderc_optimization_level opt = shaderc_optimization_level_zero
    );

    rad::Ref<Device> m_device;
    vk::raii::Pipeline m_wrapper = { nullptr };
    vk::raii::PipelineLayout m_layout = { nullptr };

    vk::PipelineBindPoint m_bindPoint;

}; // class Pipeline

class GraphicsPipeline : public Pipeline
{
public:
    GraphicsPipeline(rad::Ref<Device> device) :
        Pipeline(std::move(device), vk::PipelineBindPoint::eGraphics) {}
    GraphicsPipeline(rad::Ref<Device> device,
        const vk::GraphicsPipelineCreateInfo& createInfo,
        vk::Optional<const vk::raii::PipelineCache> cache);
    ~GraphicsPipeline();

    std::map<vk::ShaderStageFlagBits, rad::Ref<ShaderStageInfo>> m_shaderStages;

}; // class GraphicsPipeline

class ComputePipeline : public Pipeline
{
public:
    ComputePipeline(rad::Ref<Device> device) :
        Pipeline(std::move(device), vk::PipelineBindPoint::eGraphics) {}
    ComputePipeline(rad::Ref<Device> device,
        const vk::ComputePipelineCreateInfo& createInfo,
        vk::Optional<const vk::raii::PipelineCache> cache);
    ~ComputePipeline();

    rad::Ref<ShaderStageInfo> m_shaderStage;

}; // class ComputePipeline

} // namespace vkpp
