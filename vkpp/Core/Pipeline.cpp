#include <vkpp/Core/Pipeline.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/ShaderCompiler.h>

namespace vkpp
{

ShaderModule::ShaderModule(rad::Ref<Device> device, const vk::ShaderModuleCreateInfo& createInfo) :
    m_device(std::move(device))
{
    m_wrapper = m_device->m_wrapper.createShaderModule(createInfo);
}

ShaderModule::~ShaderModule()
{
}

rad::Ref<ShaderStageInfo> ShaderStageInfo::CreateFromGLSL(
    rad::Ref<Device> device, vk::ShaderStageFlagBits stage,
    const std::string& fileName, const std::string& source,
    const std::string& entryPoint, rad::Span<ShaderMacro> macros)
{
    rad::Ref<ShaderStageInfo> shaderStage;
    ShaderCompiler compiler;
    std::vector<uint32_t> code = compiler.CompileGLSL(stage, fileName, source, entryPoint, macros);
    if (!code.empty())
    {
        shaderStage = RAD_NEW ShaderStageInfo();
        shaderStage->m_stage = stage;
        shaderStage->m_module = device->CreateShaderModule(code);
        shaderStage->m_entryPoint = "main";
    }
    else
    {
        VKPP_LOG(err, "Failed to compile {}:\n{}", fileName, compiler.GetLog());
        return nullptr;
    }
    return shaderStage;
}

PipelineLayout::PipelineLayout(rad::Ref<Device> device, const vk::PipelineLayoutCreateInfo& createInfo) :
    m_device(std::move(device))
{
    m_wrapper = m_device->m_wrapper.createPipelineLayout(createInfo);
}

PipelineLayout::~PipelineLayout()
{
}

rad::Ref<ShaderStageInfo> Pipeline::CreateShaderStageFromGLSL(
    vk::ShaderStageFlagBits stage, const std::string& fileName, const std::string& source,
    const std::string& entryPoint, rad::Span<ShaderMacro> macros,
    shaderc_optimization_level opt)
{
    ShaderCompiler compiler;
    std::vector<uint32_t> code = compiler.CompileGLSL(stage, fileName, source, entryPoint, macros, opt);
    if (!code.empty())
    {
        rad::Ref<ShaderStageInfo> shaderStage = RAD_NEW ShaderStageInfo();
        shaderStage->m_stage = stage;
        shaderStage->m_module = m_device->CreateShaderModule(code);
        shaderStage->m_entryPoint = "main";

        return shaderStage;
    }
    else
    {
        VKPP_LOG(err, "Failed to compile {}:\n{}", fileName, compiler.GetLog());
        return nullptr;
    }
}

GraphicsPipeline::GraphicsPipeline(
    rad::Ref<Device> device, const vk::GraphicsPipelineCreateInfo& createInfo,
    vk::Optional<const vk::raii::PipelineCache> cache) :
    Pipeline(std::move(device), vk::PipelineBindPoint::eGraphics)
{
    m_wrapper = m_device->m_wrapper.createGraphicsPipeline(cache, createInfo, nullptr);
}

GraphicsPipeline::~GraphicsPipeline()
{
}

ComputePipeline::ComputePipeline(
    rad::Ref<Device> device, const vk::ComputePipelineCreateInfo& createInfo,
    vk::Optional<const vk::raii::PipelineCache> cache) :
    Pipeline(std::move(device), vk::PipelineBindPoint::eCompute)
{
    m_wrapper = m_device->m_wrapper.createComputePipeline(cache, createInfo, nullptr);
}

ComputePipeline::~ComputePipeline()
{
}

} // namespace vkpp
