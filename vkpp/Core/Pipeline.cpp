#include <vkpp/Core/Pipeline.h>
#include <vkpp/Core/Device.h>

namespace vkpp
{

void Pipeline::CreateLayout(const vk::PipelineLayoutCreateInfo& createInfo)
{
    m_layout = vk::raii::PipelineLayout(m_device->m_handle, createInfo);
}

void Pipeline::CreateLayout(vk::PipelineLayoutCreateFlags flags, rad::ArrayRef<vk::DescriptorSetLayout> setLayouts,
    rad::ArrayRef<vk::PushConstantRange> pushConstantRanges)
{
    vk::PipelineLayoutCreateInfo createInfo;
    createInfo.flags = flags;
    createInfo.setSetLayouts(setLayouts);
    createInfo.setPushConstantRanges(pushConstantRanges);
    CreateLayout(createInfo);
}

rad::Ref<PipelineShaderStage> Pipeline::CreateShaderStageFromGLSL(
    vk::ShaderStageFlagBits stage, const std::string& fileName, const std::string& source,
    const std::string& entryPoint, rad::Span<ShaderMacro> macros,
    shaderc_optimization_level opt)
{
    ShaderCompiler compiler;
    std::vector<uint32_t> code = compiler.CompileGLSL(stage, fileName, source, entryPoint, macros, opt);
    if (!code.empty())
    {
        vk::ShaderModuleCreateInfo shaderModuleCreateInfo;
        shaderModuleCreateInfo.codeSize = code.size() * sizeof(uint32_t);
        shaderModuleCreateInfo.pCode = code.data();
        vk::raii::ShaderModule shaderModule = m_device->m_handle.createShaderModule(shaderModuleCreateInfo);

        rad::Ref<PipelineShaderStage> shaderStage = RAD_NEW PipelineShaderStage();
        shaderStage->m_stage = stage;
        shaderStage->m_module = std::move(shaderModule);
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
