#include <MLCore/Vulkan/VulkanTensorOp.h>
#include <MLCore/Vulkan/VulkanDevice.h>
#include <MLCore/Vulkan/VulkanContext.h>
#include <MLCore/TensorIterator.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Command.h>
#include <vkpp/Core/Fence.h>
#include <vkpp/Core/Semaphore.h>
#include <vkpp/Core/Event.h>
#include <vkpp/Core/Descriptor.h>
#include <vkpp/Core/Buffer.h>
#include <vkpp/Core/Image.h>
#include <vkpp/Core/Sampler.h>
#include <vkpp/Core/Pipeline.h>

namespace ML
{

VulkanTensorOp::VulkanTensorOp(VulkanContext* context) :
    m_context(context)
{
    vkpp::Device* device = m_context->GetDeviceImpl();
    m_cmdStream = device->CreateCommandStream(vkpp::QueueFamily::Universal);
}

VulkanTensorOp::~VulkanTensorOp()
{
}

void VulkanTensorOp::CreatePipelineLayout(size_t tensorCount,
    rad::ArrayRef<vk::PushConstantRange> pushConstantRanges)
{
    vkpp::Device* device = m_context->GetDeviceImpl();
    rad::SmallVector<vk::DescriptorSetLayoutBinding, 8> bindings;
    bindings.reserve(1 + tensorCount);
    bindings.emplace_back(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute);
    for (size_t i = 0; i < tensorCount; ++i)
    {
        // binding, type, count, stageFlags, pImmutableSamplers
        bindings.emplace_back(i + 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
    }
    m_descSetLayout = device->CreateDescriptorSetLayout(bindings);
    m_pipelineLayout = device->CreatePipelineLayout(m_descSetLayout->GetHandle(), pushConstantRanges);

    m_descPool = device->CreateDescriptorPool(1,
        {
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, tensorCount),
        });
    m_descSet = m_descPool->Allocate(m_descSetLayout->GetHandle())[0];
}

void VulkanTensorOp::SetUniforms(const void* data, size_t dataSize)
{
    vkpp::Device* device = m_context->GetDeviceImpl();
    if (!m_uniformBuffer || (m_uniformBuffer->GetSize() != dataSize))
    {
        m_uniformBuffer = vkpp::Buffer::CreateUniform(device, dataSize);
        m_descSet->UpdateBuffers(0, 0, vk::DescriptorType::eUniformBuffer, m_uniformBuffer.get());
    }
    m_uniformBuffer->Write(data, 0, dataSize);
}

void VulkanTensorOp::SetTensor(uint32_t binding, VulkanTensor* tensor)
{
    vk::DescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = tensor->m_buffer->GetHandle();
    bufferInfo.offset = tensor->m_bufferOffset;
    bufferInfo.range = tensor->m_bufferSize;
    m_descSet->UpdateBuffers(binding, 0, vk::DescriptorType::eStorageBuffer, bufferInfo);
    m_bindings[binding] = tensor;
}

std::vector<size_t> VulkanTensorOp::ExpandTensorSizeND(rad::ArrayRef<size_t> sizes, size_t n)
{
    if (n > sizes.size())
    {
        std::vector<size_t> sizesExpanded(n, 1);
        for (size_t i = 0; i < sizes.size(); ++i)
        {
            sizesExpanded[i + n - sizes.size()] = sizes[i];
        }
        return sizesExpanded;
    }
    else
    {
        return sizes;
    }
}

std::vector<size_t> VulkanTensorOp::ExpandTensorStrideND(rad::ArrayRef<size_t> strides, size_t n)
{
    if (n > strides.size())
    {
        size_t maxStride = *std::max_element(strides.begin(), strides.end());
        std::vector<size_t> stridesExpanded(n, maxStride);
        for (size_t i = 0; i < strides.size(); ++i)
        {
            stridesExpanded[i + n - strides.size()] = strides[i];
        }
        return stridesExpanded;
    }
    else
    {
        return strides;
    }
}

const char* VulkanTensorOp::GetDataTypeShaderString(DataType dataType)
{
    switch (dataType)
    {
    case DataType::Float16: return "float16_t";
    case DataType::Float32: return "float32_t";
    case DataType::Float64: return "float64_t";
    case DataType::Sint8: return "int8_t";
    case DataType::Sint16: return "int16_t";
    case DataType::Sint32: return "int32_t";
    case DataType::Sint64: return "int64_t";
    case DataType::Uint8: return "uint8_t";
    case DataType::Uint16: return "uint16_t";
    case DataType::Uint32: return "uint32_t";
    case DataType::Uint64: return "uint64_t";
    case DataType::BFloat16: return "bfloat16_t";     // GL_EXT_bfloat16: https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GL_EXT_bfloat16.txt
    case DataType::Float8E4M3: return "floate4m3_t";  // GL_EXT_float_e4m3: https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GL_EXT_float8_e5m2_e4m3.txt
    case DataType::Float8E5M2: return "floate5m2_t";  // GL_EXT_float_e5m2: https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GL_EXT_float8_e5m2_e4m3.txt
    }
    RAD_UNREACHABLE();
    return nullptr;
}

VulkanTensorOpForEach::VulkanTensorOpForEach(VulkanContext* context, std::string_view opName) :
    VulkanTensorOp(context),
    m_opName(opName)
{
    vkpp::Device* device = m_context->GetDeviceImpl();

    for (size_t i = 1; i < rad::ToUnderlying(DataType::Count); ++i)
    {
        DataType dataType = static_cast<DataType>(i);
        DataType computeType = dataType;
        if (dataType == DataType::BFloat16)
        {
            computeType = DataType::Float32;
        }
        else if ((dataType == DataType::Float8E4M3) || (dataType == DataType::Float8E5M2))
        {
            computeType = DataType::Float32;
        }

        if (!m_context->GetDevice()->IsDataTypeSupported(static_cast<DataType>(i)))
        {
            continue;
        }

        std::string sourceRoot = "./Shaders/";
        const char* env = std::getenv("MLCORE_VULKAN_SHADER_SOURCE_DIR");
        if (env && rad::Exists(rad::MakeFilePath(env)))
        {
            sourceRoot = env;
        }
        std::string sourceName = sourceRoot + "/TensorOp/ForEach.comp";
        std::string source = rad::File::ReadAll(sourceName);

        std::vector<vkpp::ShaderMacro> macros =
        {
            vkpp::ShaderMacro{ "DATA_TYPE_ID", std::to_string(rad::ToUnderlying(dataType)) },
            vkpp::ShaderMacro{ "DATA_TYPE", GetDataTypeShaderString(dataType) },
            vkpp::ShaderMacro{ "COMPUTE_TYPE_ID", std::to_string(rad::ToUnderlying(computeType)) },
            vkpp::ShaderMacro{ "COMPUTE_TYPE", GetDataTypeShaderString(computeType) },
            vkpp::ShaderMacro{ "OP_NAME", m_opName },
        };
        rad::Ref<vkpp::ShaderStageInfo> shaderStage = vkpp::ShaderStageInfo::CreateFromGLSL(
            device, vk::ShaderStageFlagBits::eCompute, sourceName,
            source, "main", macros
        );

        vk::PushConstantRange pushConstantRange = {};
        pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstants);
        CreatePipelineLayout(1, pushConstantRange);
        m_pipelines[dataType] = device->CreateComputePipeline(shaderStage, m_pipelineLayout->GetHandle());
        m_pipelines[dataType]->m_layout = m_pipelineLayout;
    }
}

VulkanTensorOpForEach::~VulkanTensorOpForEach()
{
}

void VulkanTensorOpForEach::Execute()
{
    vkpp::Device* device = m_context->GetDeviceImpl();

    VulkanTensor* input = GetInputTensor();

    DataType dataType = input->m_dataType;

    std::vector<size_t> dispatchSizes = ExpandTensorSizeND(input->m_sizes, DispatchDimCount);
    std::vector<size_t> dispatchStrides = ExpandTensorStrideND(input->m_strides, DispatchDimCount);

    glm::uvec3 m_threadGroupCount;
    m_threadGroupCount.x = rad::DivRoundUp<uint32_t>(static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 1]), 16u);  // W
    m_threadGroupCount.y = rad::DivRoundUp<uint32_t>(static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 2]), 16u);  // H
    m_threadGroupCount.z = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 3]);

    m_uniformData.sizes[0] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 4]);
    m_uniformData.sizes[1] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 3]);
    m_uniformData.sizes[2] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 2]);
    m_uniformData.sizes[3] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 1]);
    m_uniformData.strides[0] = static_cast<uint32_t>(dispatchStrides[DispatchDimCount - 4]);
    m_uniformData.strides[1] = static_cast<uint32_t>(dispatchStrides[DispatchDimCount - 3]);
    m_uniformData.strides[2] = static_cast<uint32_t>(dispatchStrides[DispatchDimCount - 2]);
    m_uniformData.strides[3] = static_cast<uint32_t>(dispatchStrides[DispatchDimCount - 1]);
    SetUniforms(m_uniformData);

    rad::Ref<vkpp::CommandBuffer> cmdBuffer = m_cmdStream->m_cmdPoolTransientAlloc->AllocatePrimary();

    cmdBuffer->Begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    auto& memoryBarriers = m_context->m_memoryBarriers;
    if (memoryBarriers.size() > 0)
    {
        vk::DependencyInfoKHR dependency;
        dependency.setMemoryBarriers(memoryBarriers);
        cmdBuffer->SetPipelineBarrier2(dependency);
        memoryBarriers.clear();
    }

    // Select the best pipeline according to data type, input/output sizes and strides.
    vkpp::Pipeline* pipeline = m_pipelines[dataType].get();
    cmdBuffer->BindPipeline(vk::PipelineBindPoint::eCompute, pipeline->m_wrapper);
    cmdBuffer->BindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipelineLayout->GetHandle(), 0,
        { m_descSet->GetHandle() }, {});

    TensorIterator iter(input);
    do {
        PushConstants pushConstants = {};
        const auto& offsets = iter.m_coord;
        pushConstants.inputIndexOffset = input->CoordToBufferIndex(offsets);
        cmdBuffer->SetPushConstants<PushConstants>(
            m_pipelineLayout->GetHandle(), vk::ShaderStageFlagBits::eCompute, 0, pushConstants);
        cmdBuffer->Dispatch(m_threadGroupCount.x, m_threadGroupCount.y, m_threadGroupCount.z);
    } while (iter.NextND(DispatchDimCount));

    cmdBuffer->End();

    m_cmdStream->SubmitAndWaitForCompletion(cmdBuffer->GetHandle(),
        m_context->m_submitWaits, m_context->m_submitSignals);
    m_context->m_submitWaits.clear();
    m_context->m_submitSignals.clear();
}


VulkanTensorOpElementWiseUnary::VulkanTensorOpElementWiseUnary(VulkanContext* context, std::string_view opName) :
    VulkanTensorOp(context),
    m_opName(opName)
{
    vkpp::Device* device = m_context->GetDeviceImpl();

    for (size_t i = 1; i < rad::ToUnderlying(DataType::Count); ++i)
    {
        DataType dataType = static_cast<DataType>(i);
        DataType computeType = dataType;
        if (dataType == DataType::BFloat16)
        {
            computeType = DataType::Float32;
        }
        else if ((dataType == DataType::Float8E4M3) || (dataType == DataType::Float8E5M2))
        {
            computeType = DataType::Float32;
        }

        if (!m_context->GetDevice()->IsDataTypeSupported(static_cast<DataType>(i)))
        {
            continue;
        }

        std::string sourceRoot = "./Shaders/";
        const char* env = std::getenv("MLCORE_VULKAN_SHADER_SOURCE_DIR");
        if (env && rad::Exists(rad::MakeFilePath(env)))
        {
            sourceRoot = env;
        }
        std::string sourceName = sourceRoot + "/TensorOp/ElementWiseUnary.comp";
        std::string source = rad::File::ReadAll(sourceName);

        std::vector<vkpp::ShaderMacro> macros =
        {
            vkpp::ShaderMacro{ "DATA_TYPE_ID", std::to_string(rad::ToUnderlying(dataType)) },
            vkpp::ShaderMacro{ "DATA_TYPE", GetDataTypeShaderString(dataType) },
            vkpp::ShaderMacro{ "COMPUTE_TYPE_ID", std::to_string(rad::ToUnderlying(computeType)) },
            vkpp::ShaderMacro{ "COMPUTE_TYPE", GetDataTypeShaderString(computeType) },
            vkpp::ShaderMacro{ "OP_NAME", m_opName },
        };
        rad::Ref<vkpp::ShaderStageInfo> shaderStage = vkpp::ShaderStageInfo::CreateFromGLSL(
            device, vk::ShaderStageFlagBits::eCompute, sourceName,
            source, "main", macros
        );

        vk::PushConstantRange pushConstantRange = {};
        pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstants);
        CreatePipelineLayout(2, pushConstantRange);
        m_pipelines[dataType] = device->CreateComputePipeline(shaderStage, m_pipelineLayout->GetHandle());
        m_pipelines[dataType]->m_layout = m_pipelineLayout;
    }
}

VulkanTensorOpElementWiseUnary::~VulkanTensorOpElementWiseUnary()
{
}

void VulkanTensorOpElementWiseUnary::Execute()
{
    vkpp::Device* device = m_context->GetDeviceImpl();

    VulkanTensor* input = GetInputTensor();
    VulkanTensor* output = GetOutputTensor();
    assert(output->m_sizes == input->m_sizes);
    assert(output->m_dataType == input->m_dataType);

    DataType dataType = input->m_dataType;

    std::vector<size_t> dispatchSizes = ExpandTensorSizeND(input->m_sizes, DispatchDimCount);
    std::vector<size_t> dispatchInputStrides = ExpandTensorStrideND(input->m_strides, DispatchDimCount);
    std::vector<size_t> dispatchOutputStrides = ExpandTensorStrideND(output->m_strides, DispatchDimCount);

    glm::uvec3 m_threadGroupCount;
    m_threadGroupCount.x = rad::DivRoundUp<uint32_t>(static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 1]), 16u);  // W
    m_threadGroupCount.y = rad::DivRoundUp<uint32_t>(static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 2]), 16u);  // H
    m_threadGroupCount.z = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - DispatchDimCount]);

    m_uniformData.sizes[0] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 4]);
    m_uniformData.sizes[1] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 3]);
    m_uniformData.sizes[2] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 2]);
    m_uniformData.sizes[3] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 1]);
    m_uniformData.inputStrides[0] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 4]);
    m_uniformData.inputStrides[1] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 3]);
    m_uniformData.inputStrides[2] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 2]);
    m_uniformData.inputStrides[3] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 1]);
    m_uniformData.outputStrides[0] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 4]);
    m_uniformData.outputStrides[1] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 3]);
    m_uniformData.outputStrides[2] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 2]);
    m_uniformData.outputStrides[3] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 1]);
    SetUniforms(m_uniformData);

    rad::Ref<vkpp::CommandBuffer> cmdBuffer = m_cmdStream->m_cmdPoolTransientAlloc->AllocatePrimary();

    cmdBuffer->Begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    auto& memoryBarriers = m_context->m_memoryBarriers;
    if (memoryBarriers.size() > 0)
    {
        vk::DependencyInfoKHR dependency;
        dependency.setMemoryBarriers(memoryBarriers);
        cmdBuffer->SetPipelineBarrier2(dependency);
        memoryBarriers.clear();
    }

    // Select the best pipeline according to data type, input/output sizes and strides.
    vkpp::Pipeline* pipeline = m_pipelines[dataType].get();
    cmdBuffer->BindPipeline(vk::PipelineBindPoint::eCompute, pipeline->m_wrapper);
    cmdBuffer->BindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipelineLayout->GetHandle(), 0,
        { m_descSet->GetHandle() }, {});

    TensorIterator iter(input);
    do {
        PushConstants pushConstants = {};
        const auto& offsets = iter.m_coord;
        pushConstants.inputIndexOffset = input->CoordToBufferIndex(offsets);
        pushConstants.outputIndexOffset = output->CoordToBufferIndex(offsets);
        cmdBuffer->SetPushConstants<PushConstants>(
            m_pipelineLayout->GetHandle(), vk::ShaderStageFlagBits::eCompute, 0, pushConstants);
        cmdBuffer->Dispatch(m_threadGroupCount.x, m_threadGroupCount.y, m_threadGroupCount.z);
    } while (iter.NextND(DispatchDimCount));

    cmdBuffer->End();

    m_cmdStream->SubmitAndWaitForCompletion(cmdBuffer->GetHandle(),
        m_context->m_submitWaits, m_context->m_submitSignals);
    m_context->m_submitWaits.clear();
    m_context->m_submitSignals.clear();
}

VulkanTensorOpElementWiseBinary::VulkanTensorOpElementWiseBinary(VulkanContext* context, std::string_view opName) :
    VulkanTensorOp(context),
    m_opName(opName)
{
    vkpp::Device* device = m_context->GetDeviceImpl();

    for (size_t i = 1; i < rad::ToUnderlying(DataType::Count); ++i)
    {
        DataType dataType = static_cast<DataType>(i);
        DataType computeType = dataType;
        if (dataType == DataType::BFloat16)
        {
            computeType = DataType::Float32;
        }
        else if ((dataType == DataType::Float8E4M3) || (dataType == DataType::Float8E5M2))
        {
            computeType = DataType::Float32;
        }

        if (!m_context->GetDevice()->IsDataTypeSupported(static_cast<DataType>(i)))
        {
            continue;
        }

        std::string sourceRoot = "./Shaders/";
        const char* env = std::getenv("MLCORE_VULKAN_SHADER_SOURCE_DIR");
        if (env && rad::Exists(rad::MakeFilePath(env)))
        {
            sourceRoot = env;
        }
        std::string sourceName = sourceRoot + "/TensorOp/ElementWiseBinary.comp";
        std::string source = rad::File::ReadAll(sourceName);

        std::vector<vkpp::ShaderMacro> macros =
        {
            vkpp::ShaderMacro{ "DATA_TYPE_ID", std::to_string(rad::ToUnderlying(dataType)) },
            vkpp::ShaderMacro{ "DATA_TYPE", GetDataTypeShaderString(dataType) },
            vkpp::ShaderMacro{ "COMPUTE_TYPE_ID", std::to_string(rad::ToUnderlying(computeType)) },
            vkpp::ShaderMacro{ "COMPUTE_TYPE", GetDataTypeShaderString(computeType) },
            vkpp::ShaderMacro{ "OP_NAME", m_opName },
        };
        rad::Ref<vkpp::ShaderStageInfo> shaderStage = vkpp::ShaderStageInfo::CreateFromGLSL(
            device, vk::ShaderStageFlagBits::eCompute, sourceName,
            source, "main", macros
        );

        vk::PushConstantRange pushConstantRange = {};
        pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstants);
        CreatePipelineLayout(3, pushConstantRange);
        m_pipelines[dataType] = device->CreateComputePipeline(shaderStage, m_pipelineLayout->GetHandle());
        m_pipelines[dataType]->m_layout = m_pipelineLayout;
    }
}

VulkanTensorOpElementWiseBinary::~VulkanTensorOpElementWiseBinary()
{
}

void VulkanTensorOpElementWiseBinary::Execute()
{
    vkpp::Device* device = m_context->GetDeviceImpl();

    VulkanTensor* input = GetInputTensor();
    VulkanTensor* other = GetInputTensor();
    VulkanTensor* output = GetOutputTensor();
    assert(other->m_sizes == input->m_sizes);
    assert(other->m_dataType == input->m_dataType);
    assert(output->m_sizes == input->m_sizes);
    assert(output->m_dataType == input->m_dataType);

    DataType dataType = input->m_dataType;

    std::vector<size_t> dispatchSizes = ExpandTensorSizeND(input->m_sizes, DispatchDimCount);
    std::vector<size_t> dispatchInputStrides = ExpandTensorStrideND(input->m_strides, DispatchDimCount);
    std::vector<size_t> dispatchOtherStrides = ExpandTensorStrideND(input->m_strides, DispatchDimCount);
    std::vector<size_t> dispatchOutputStrides = ExpandTensorStrideND(output->m_strides, DispatchDimCount);

    glm::uvec3 m_threadGroupCount;
    m_threadGroupCount.x = rad::DivRoundUp<uint32_t>(static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 1]), 16u);  // W
    m_threadGroupCount.y = rad::DivRoundUp<uint32_t>(static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 2]), 16u);  // H
    m_threadGroupCount.z = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - DispatchDimCount]);

    m_uniformData.sizes[0] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 4]);
    m_uniformData.sizes[1] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 3]);
    m_uniformData.sizes[2] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 2]);
    m_uniformData.sizes[3] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 1]);
    m_uniformData.inputStrides[0] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 4]);
    m_uniformData.inputStrides[1] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 3]);
    m_uniformData.inputStrides[2] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 2]);
    m_uniformData.inputStrides[3] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 1]);
    m_uniformData.otherStrides[0] = static_cast<uint32_t>(dispatchOtherStrides[DispatchDimCount - 4]);
    m_uniformData.otherStrides[1] = static_cast<uint32_t>(dispatchOtherStrides[DispatchDimCount - 3]);
    m_uniformData.otherStrides[2] = static_cast<uint32_t>(dispatchOtherStrides[DispatchDimCount - 2]);
    m_uniformData.otherStrides[3] = static_cast<uint32_t>(dispatchOtherStrides[DispatchDimCount - 1]);
    m_uniformData.outputStrides[0] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 4]);
    m_uniformData.outputStrides[1] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 3]);
    m_uniformData.outputStrides[2] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 2]);
    m_uniformData.outputStrides[3] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 1]);
    SetUniforms(m_uniformData);

    rad::Ref<vkpp::CommandBuffer> cmdBuffer = m_cmdStream->m_cmdPoolTransientAlloc->AllocatePrimary();

    cmdBuffer->Begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    auto& memoryBarriers = m_context->m_memoryBarriers;
    if (memoryBarriers.size() > 0)
    {
        vk::DependencyInfoKHR dependency;
        dependency.setMemoryBarriers(memoryBarriers);
        cmdBuffer->SetPipelineBarrier2(dependency);
        memoryBarriers.clear();
    }

    // Select the best pipeline according to data type, input/output sizes and strides.
    vkpp::Pipeline* pipeline = m_pipelines[dataType].get();
    cmdBuffer->BindPipeline(vk::PipelineBindPoint::eCompute, pipeline->m_wrapper);
    cmdBuffer->BindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipelineLayout->GetHandle(), 0,
        { m_descSet->GetHandle() }, {});

    TensorIterator iter(input);
    do {
        PushConstants pushConstants = {};
        const auto& offsets = iter.m_coord;
        pushConstants.inputIndexOffset = input->CoordToBufferIndex(offsets);
        pushConstants.outputIndexOffset = output->CoordToBufferIndex(offsets);
        cmdBuffer->SetPushConstants<PushConstants>(
            m_pipelineLayout->GetHandle(), vk::ShaderStageFlagBits::eCompute, 0, pushConstants);
        cmdBuffer->Dispatch(m_threadGroupCount.x, m_threadGroupCount.y, m_threadGroupCount.z);
    } while (iter.NextND(DispatchDimCount));

    cmdBuffer->End();

    m_cmdStream->SubmitAndWaitForCompletion(cmdBuffer->GetHandle(),
        m_context->m_submitWaits, m_context->m_submitSignals);
    m_context->m_submitWaits.clear();
    m_context->m_submitSignals.clear();
}

} // namespace ML
