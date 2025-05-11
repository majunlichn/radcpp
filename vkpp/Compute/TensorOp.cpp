#include <vkpp/Compute/TensorOp.h>
#include <vkpp/Core/Command.h>

namespace vkpp
{

TensorOp::TensorOp(rad::Ref<Device> device) :
    m_device(std::move(device))
{
}

TensorOp::~TensorOp()
{
}

const char* TensorOp::GetDataTypeShaderString(vk::ComponentTypeKHR dataType)
{
    switch (dataType)
    {
    case vk::ComponentTypeKHR::eFloat16: return "float16_t";
    case vk::ComponentTypeKHR::eFloat32: return "float32_t";
    case vk::ComponentTypeKHR::eFloat64: return "float64_t";
    case vk::ComponentTypeKHR::eSint8: return "int8_t";
    case vk::ComponentTypeKHR::eSint16: return "int16_t";
    case vk::ComponentTypeKHR::eSint32: return "int32_t";
    case vk::ComponentTypeKHR::eSint64: return "int64_t";
    case vk::ComponentTypeKHR::eUint8: return "uint8_t";
    case vk::ComponentTypeKHR::eUint16: return "uint16_t";
    case vk::ComponentTypeKHR::eUint32: return "uint32_t";
    case vk::ComponentTypeKHR::eUint64: return "uint64_t";
    }
    return nullptr;
}

void TensorOp::CreatePipelineLayouts(size_t tensorCount,
    rad::ArrayRef<vk::PushConstantRange> pushConstantRanges)
{
    rad::SmallVector<vk::DescriptorSetLayoutBinding, 8> bindings;
    bindings.reserve(1 + tensorCount);
    bindings.emplace_back(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute);
    for (size_t i = 0; i < tensorCount; ++i)
    {
        // binding, type, count, stageFlags, pImmutableSamplers
        bindings.emplace_back(i + 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
    }
    m_descSetLayout = m_device->CreateDescriptorSetLayout(bindings);
    m_pipelineLayout = m_device->CreatePipelineLayout({ m_descSetLayout }, pushConstantRanges);

    m_descPool = m_device->CreateDescriptorPool(1,
        {
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, tensorCount),
        });
    m_descSets = m_descPool->Allocate({ m_descSetLayout });
}

void TensorOp::SetUniforms(const void* data, size_t dataSize)
{
    if (!m_uniformBuffer || (m_uniformBuffer->GetSize() != dataSize))
    {
        m_uniformBuffer = Buffer::CreateUniform(m_device, dataSize);
        DescriptorUpdater(m_descSets[0])
            .UpdateBuffers(0, 0, vk::DescriptorType::eUniformBuffer, m_uniformBuffer.get());
    }
    m_uniformBuffer->Write(data, 0, dataSize);
}

void TensorOp::SetTensor(uint32_t binding, Tensor* tensor)
{
    DescriptorUpdater updater(m_descSets[0]);
    vk::DescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = tensor->m_buffer->GetHandle();
    bufferInfo.offset = tensor->m_bufferOffset;
    bufferInfo.range = tensor->m_sizeInBytes;
    updater.UpdateBuffers(binding, 0, vk::DescriptorType::eStorageBuffer, bufferInfo);
    m_bindings[binding] = tensor;
}

void TensorOp::Execute(glm::uvec3 groupCount)
{
    vk::raii::CommandBuffer cmdBuffer = m_device->AllocateTemporaryCommandBuffer(QueueFamily::Universal);

    CommandRecorder(cmdBuffer).Begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    if (m_enable_PreExecute_MemoryBarrierRAW)
    {
        CommandRecorder(cmdBuffer).SetMemoryBarrier_ComputeToComputeRAW();
    }

    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline->m_handle);
    cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipelineLayout, 0,
        { m_descSets[0] }, {});
    cmdBuffer.dispatch(groupCount.x, groupCount.y, groupCount.z);
    CommandRecorder(cmdBuffer).End();

    m_device->ExecuteSync(m_executeWaits, { cmdBuffer }, m_executeSignalSemaphores);
}

TensorOpElementWiseUnary::TensorOpElementWiseUnary(rad::Ref<Device> device) :
    TensorOp(std::move(device))
{
}

TensorOpElementWiseUnary::~TensorOpElementWiseUnary()
{
}

bool TensorOpElementWiseUnary::Init(const TensorOpElementWiseUnaryDesc& desc)
{
    m_desc = desc;

    std::string sourceRoot;
    const char* env = std::getenv("VKPP_SHADERS_ROOT");
    if (env && rad::Exists(rad::MakeFilePath(env)))
    {
        sourceRoot = env;
    }
    std::string sourceName = sourceRoot + "/TensorOp/ElementWiseUnary4D.comp";
    std::string source = rad::File::ReadAll(sourceName);
    std::vector<ShaderMacro> macros =
    {
        ShaderMacro{ "DATA_TYPE_ID", std::to_string(rad::ToUnderlying(m_desc.dataType)) },
        ShaderMacro{ "DATA_TYPE", GetDataTypeShaderString(m_desc.dataType) },
        ShaderMacro{ "OP_NAME", m_desc.opName },
    };
    rad::Ref<ShaderStageInfo> shaderStage = ShaderStageInfo::CreateFromGLSL(
        m_device, vk::ShaderStageFlagBits::eCompute, sourceName,
        source, "main", macros
    );

    CreatePipelineLayouts(2);
    m_pipeline = m_device->CreateComputePipeline(shaderStage, m_pipelineLayout);

    UpdateUniforms();

    return true;
}

void TensorOpElementWiseUnary::UpdateUniforms()
{
    if (m_desc.sizes.size() == 4)
    {
        m_uniforms.sizes[0] = static_cast<uint32_t>(m_desc.sizes[0]);
        m_uniforms.sizes[1] = static_cast<uint32_t>(m_desc.sizes[1]);
        m_uniforms.sizes[2] = static_cast<uint32_t>(m_desc.sizes[2]);
        m_uniforms.sizes[3] = static_cast<uint32_t>(m_desc.sizes[3]);
        m_uniforms.inputStrides[0] = static_cast<uint32_t>(m_desc.inputStrides[0]);
        m_uniforms.inputStrides[1] = static_cast<uint32_t>(m_desc.inputStrides[1]);
        m_uniforms.inputStrides[2] = static_cast<uint32_t>(m_desc.inputStrides[2]);
        m_uniforms.inputStrides[3] = static_cast<uint32_t>(m_desc.inputStrides[3]);
        m_uniforms.outputStrides[0] = static_cast<uint32_t>(m_desc.outputStrides[0]);
        m_uniforms.outputStrides[1] = static_cast<uint32_t>(m_desc.outputStrides[1]);
        m_uniforms.outputStrides[2] = static_cast<uint32_t>(m_desc.outputStrides[2]);
        m_uniforms.outputStrides[3] = static_cast<uint32_t>(m_desc.outputStrides[3]);
    }
    SetUniforms(m_uniforms);
}

} // namespace vkpp
