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
    ShaderCompiler compiler;
    std::vector<uint32_t> code = compiler.CompileGLSL(stage, fileName, source, entryPoint, macros);
    if (!code.empty())
    {
        rad::Ref<ShaderStageInfo> shaderStage = RAD_NEW ShaderStageInfo();
        shaderStage->m_stage = stage;
        shaderStage->m_module = device->CreateShaderModule(code);
        shaderStage->m_entryPoint = "main";
        return shaderStage;
    }
    else
    {
        VKPP_LOG(err, "Failed to compile {}:\n{}", fileName, compiler.GetLog());
        return nullptr;
    }
}

rad::Ref<ShaderStageInfo> ShaderStageInfo::CreateFromCompiledBinary(rad::Ref<Device> device, vk::ShaderStageFlagBits stage, rad::ArrayRef<uint32_t> code)
{
    rad::Ref<ShaderStageInfo> shaderStage = RAD_NEW ShaderStageInfo();
    shaderStage->m_stage = stage;
    shaderStage->m_module = device->CreateShaderModule(code);
    shaderStage->m_entryPoint = "main";
    return shaderStage;
}

rad::Ref<ShaderStageInfo> ShaderStageInfo::CreateFromCompiledBinaryFile(rad::Ref<Device> device, vk::ShaderStageFlagBits stage, const std::string& fileName)
{
    rad::File file;
    if (file.Open(fileName, "rb"))
    {
        size_t fileSize = static_cast<size_t>(file.GetSize());
        assert((fileSize % 4) == 0);
        std::vector<uint32_t> code(fileSize / sizeof(uint32_t));
        file.Read(code.data(), fileSize);
        return CreateFromCompiledBinary(std::move(device), stage, code);
    }
    else
    {
        VKPP_LOG(err, "Cannot open file: {}", fileName);
        return nullptr;
    }
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
