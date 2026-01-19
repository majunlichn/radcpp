#include <MLCore/Vulkan/VulkanTensorOp.h>
#include <MLCore/Vulkan/VulkanDevice.h>
#include <MLCore/Vulkan/VulkanContext.h>
#include <MLCore/TensorIterator.h>
#include <MLCore/Logging.h>
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
    vkpp::Device* device = m_context->GetDevice()->m_impl.get();
    m_cmdStream = device->CreateCommandStream(vkpp::QueueFamily::Universal);
}

VulkanTensorOp::~VulkanTensorOp()
{
}

void VulkanTensorOp::CreatePipelineLayout(size_t tensorCount,
    rad::ArrayRef<vk::PushConstantRange> pushConstantRanges)
{
    vkpp::Device* device = m_context->GetDevice()->m_impl.get();
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
    vkpp::Device* device = m_context->GetDevice()->m_impl.get();
    if (!m_uniformBuffer || (m_uniformBuffer->GetSize() != dataSize))
    {
        m_uniformBuffer = vkpp::Buffer::CreateUniform(device, dataSize);
        m_descSet->UpdateBuffers(0, 0, vk::DescriptorType::eUniformBuffer, m_uniformBuffer.get());
    }
    m_uniformBuffer->Write(data, 0, dataSize);
}

void VulkanTensorOp::SetTensor(uint32_t binding, const Tensor& tensor)
{
    VulkanTensorStorage* tensorStorage = static_cast<VulkanTensorStorage*>(tensor.m_storage.get());
    vk::DescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = tensorStorage->m_buffer->GetHandle();
    bufferInfo.offset = tensor.m_bufferOffset;
    bufferInfo.range = tensor.m_bufferSize;
    m_descSet->UpdateBuffers(binding, 0, vk::DescriptorType::eStorageBuffer, bufferInfo);
    m_bindings[binding] = &tensor;
}

std::vector<size_t> VulkanTensorOp::ExpandTensorSizeND(rad::ArrayRef<size_t> sizes, size_t nd)
{
    if (nd > sizes.size())
    {
        std::vector<size_t> sizesExpanded(nd, 1);
        for (size_t i = 0; i < sizes.size(); ++i)
        {
            sizesExpanded[i + nd - sizes.size()] = sizes[i];
        }
        return sizesExpanded;
    }
    else
    {
        return sizes;
    }
}

std::vector<size_t> VulkanTensorOp::ExpandTensorStrideND(rad::ArrayRef<size_t> strides, size_t nd)
{
    if (nd > strides.size())
    {
        size_t maxStride = *std::max_element(strides.begin(), strides.end());
        std::vector<size_t> stridesExpanded(nd, maxStride);
        for (size_t i = 0; i < strides.size(); ++i)
        {
            stridesExpanded[i + nd - strides.size()] = strides[i];
        }
        return stridesExpanded;
    }
    else
    {
        return strides;
    }
}

std::string VulkanTensorOp::GetShaderBinaryDir() const
{
    std::string shaderBinaryDir = "Shaders/";
    const char* shaderBinaryEnv = std::getenv("MLCORE_VULKAN_SHADERS");
    if (shaderBinaryEnv)
    {
        auto path = rad::MakeFilePath(shaderBinaryEnv);
        if (rad::Exists(path))
        {
            shaderBinaryDir = (const char*)rad::MakeAbsolute(path).u8string().c_str();
        }
        else
        {
            ML_LOG(info, "MLCORE_VULKAN_SHADERS={}: the path doesn't exist!", shaderBinaryEnv);
        }
    }
    return shaderBinaryDir;
}

void VulkanElementWiseParams::Set(DataType dataType, const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d)
{
    switch (dataType)
    {
    case ML::DataType::Float16:
    case ML::DataType::Float32:
        m_params.f32 = glm::f32vec4(
            static_cast<rad::Float32>(a), static_cast<rad::Float32>(b),
            static_cast<rad::Float32>(c), static_cast<rad::Float32>(d));
        return;
    case ML::DataType::Float64:
        m_params.f64 = glm::f64vec4(
            static_cast<rad::Float64>(a), static_cast<rad::Float64>(b),
            static_cast<rad::Float64>(c), static_cast<rad::Float64>(d));
        return;
    case ML::DataType::Sint8:
    case ML::DataType::Sint16:
    case ML::DataType::Sint32:
        m_params.i32 = glm::i32vec4(
            static_cast<int32_t>(a), static_cast<int32_t>(b),
            static_cast<int32_t>(c), static_cast<int32_t>(d));
        return;
    case ML::DataType::Sint64:
        m_params.i64 = glm::i64vec4(
            static_cast<int64_t>(a), static_cast<int64_t>(b),
            static_cast<int64_t>(c), static_cast<int64_t>(d));
        return;
    case ML::DataType::Uint8:
    case ML::DataType::Uint16:
    case ML::DataType::Uint32:
        m_params.u32 = glm::u32vec4(
            static_cast<uint32_t>(a), static_cast<uint32_t>(b),
            static_cast<uint32_t>(c), static_cast<uint32_t>(d));
        return;
    case ML::DataType::Uint64:
        m_params.u64 = glm::u64vec4(
            static_cast<uint64_t>(a), static_cast<uint64_t>(b),
            static_cast<uint64_t>(c), static_cast<uint64_t>(d));
        return;
    case ML::DataType::Bool:
        m_params.u32 = glm::u32vec4(
            static_cast<bool>(a), static_cast<bool>(b),
            static_cast<bool>(c), static_cast<bool>(d));
        return;
    case ML::DataType::Complex32:
        m_params.c32[0] = static_cast<rad::Complex32>(a);
        m_params.c32[1] = static_cast<rad::Complex32>(b);
        m_params.c32[2] = static_cast<rad::Complex32>(c);
        m_params.c32[3] = static_cast<rad::Complex32>(d);
        return;
    case ML::DataType::Complex64:
        m_params.c64[0] = static_cast<rad::Complex64>(a);
        m_params.c64[1] = static_cast<rad::Complex64>(b);
        m_params.c64[2] = static_cast<rad::Complex64>(c);
        m_params.c64[3] = static_cast<rad::Complex64>(d);
        return;
    case ML::DataType::Complex128:
        m_params.c128[0] = static_cast<rad::Complex128>(a);
        m_params.c128[1] = static_cast<rad::Complex128>(b);
        m_params.c128[2] = static_cast<rad::Complex128>(c);
        m_params.c128[3] = static_cast<rad::Complex128>(d);
        return;
    }
    RAD_UNREACHABLE();
}

VulkanTensorOpForEach::VulkanTensorOpForEach(VulkanContext* context, std::string_view opName) :
    VulkanTensorOp(context),
    m_opName(opName)
{
    vkpp::Device* device = m_context->GetDevice()->m_impl.get();

    for (size_t i = 1; i < rad::ToUnderlying(DataType::Count); ++i)
    {
        DataType dataType = static_cast<DataType>(i);
        if (!m_context->GetDevice()->IsDataTypeSupported(dataType))
        {
            continue;
        }
        std::string shaderBinaryPath = GetShaderBinaryDir() + "/TensorOp/" + std::string(opName) + "-" + GetDataTypeName(dataType) + ".spv";
        rad::Ref<vkpp::ShaderStageInfo> shaderStage = vkpp::ShaderStageInfo::CreateFromCompiledBinaryFile(
            device, vk::ShaderStageFlagBits::eCompute, shaderBinaryPath
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
    vkpp::Device* device = m_context->GetDevice()->m_impl.get();

    const Tensor* input = GetInputTensor();

    DataType dataType = input->m_dataType;

    std::vector<size_t> dispatchSizes = ExpandTensorSizeND(input->m_sizes, DispatchDimCount);
    std::vector<size_t> dispatchStrides = ExpandTensorStrideND(input->m_strides, DispatchDimCount);

    glm::uvec3 m_threadGroupCount;
    m_threadGroupCount.x = rad::DivRoundUp<uint32_t>(static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 1]), 16u);  // W
    m_threadGroupCount.y = rad::DivRoundUp<uint32_t>(static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 2]), 16u);  // H
    m_threadGroupCount.z = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 3]);

    m_shaderUniforms.sizes[0] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 4]);
    m_shaderUniforms.sizes[1] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 3]);
    m_shaderUniforms.sizes[2] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 2]);
    m_shaderUniforms.sizes[3] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 1]);
    m_shaderUniforms.strides[0] = static_cast<uint32_t>(dispatchStrides[DispatchDimCount - 4]);
    m_shaderUniforms.strides[1] = static_cast<uint32_t>(dispatchStrides[DispatchDimCount - 3]);
    m_shaderUniforms.strides[2] = static_cast<uint32_t>(dispatchStrides[DispatchDimCount - 2]);
    m_shaderUniforms.strides[3] = static_cast<uint32_t>(dispatchStrides[DispatchDimCount - 1]);
    SetUniforms(m_shaderUniforms);

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

VulkanTensorOpElementWiseUnary::VulkanTensorOpElementWiseUnary(
    VulkanContext* context, std::string_view opName, rad::ArrayRef<DataType> dataTypes) :
    VulkanTensorOp(context),
    m_opName(opName)
{
    vkpp::Device* device = m_context->GetDevice()->m_impl.get();

    for (DataType dataType : dataTypes)
    {
        if (!m_context->GetDevice()->IsDataTypeComputable(dataType))
        {
            continue;
        }
        std::string shaderBinaryPath = GetShaderBinaryDir() + "/TensorOp/" + std::string(opName) + "-" + GetDataTypeName(dataType) + ".spv";
        rad::Ref<vkpp::ShaderStageInfo> shaderStage = vkpp::ShaderStageInfo::CreateFromCompiledBinaryFile(
            device, vk::ShaderStageFlagBits::eCompute, shaderBinaryPath
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
    vkpp::Device* device = m_context->GetDevice()->m_impl.get();

    const Tensor* input = GetInputTensor();
    const Tensor* output = GetOutputTensor();
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

    m_shaderUniforms.sizes[0] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 4]);
    m_shaderUniforms.sizes[1] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 3]);
    m_shaderUniforms.sizes[2] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 2]);
    m_shaderUniforms.sizes[3] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 1]);
    m_shaderUniforms.inputStrides[0] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 4]);
    m_shaderUniforms.inputStrides[1] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 3]);
    m_shaderUniforms.inputStrides[2] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 2]);
    m_shaderUniforms.inputStrides[3] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 1]);
    m_shaderUniforms.outputStrides[0] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 4]);
    m_shaderUniforms.outputStrides[1] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 3]);
    m_shaderUniforms.outputStrides[2] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 2]);
    m_shaderUniforms.outputStrides[3] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 1]);
    SetUniforms(m_shaderUniforms);

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

VulkanTensorOpElementWiseBinary::VulkanTensorOpElementWiseBinary(
    VulkanContext* context, std::string_view opName, rad::ArrayRef<DataType> dataTypes) :
    VulkanTensorOp(context),
    m_opName(opName)
{
    vkpp::Device* device = m_context->GetDevice()->m_impl.get();

    for (DataType dataType : dataTypes)
    {
        if (!m_context->GetDevice()->IsDataTypeComputable(dataType))
        {
            continue;
        }
        std::string shaderBinaryPath = GetShaderBinaryDir() + "/TensorOp/" + std::string(opName) + "-" + GetDataTypeName(dataType) + ".spv";
        rad::Ref<vkpp::ShaderStageInfo> shaderStage = vkpp::ShaderStageInfo::CreateFromCompiledBinaryFile(
            device, vk::ShaderStageFlagBits::eCompute, shaderBinaryPath
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
    vkpp::Device* device = m_context->GetDevice()->m_impl.get();

    const Tensor* input = GetInputTensor();
    const Tensor* other = GetInputTensor();
    const Tensor* output = GetOutputTensor();
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

    m_shaderUniforms.sizes[0] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 4]);
    m_shaderUniforms.sizes[1] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 3]);
    m_shaderUniforms.sizes[2] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 2]);
    m_shaderUniforms.sizes[3] = static_cast<uint32_t>(dispatchSizes[DispatchDimCount - 1]);
    m_shaderUniforms.inputStrides[0] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 4]);
    m_shaderUniforms.inputStrides[1] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 3]);
    m_shaderUniforms.inputStrides[2] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 2]);
    m_shaderUniforms.inputStrides[3] = static_cast<uint32_t>(dispatchInputStrides[DispatchDimCount - 1]);
    m_shaderUniforms.otherStrides[0] = static_cast<uint32_t>(dispatchOtherStrides[DispatchDimCount - 4]);
    m_shaderUniforms.otherStrides[1] = static_cast<uint32_t>(dispatchOtherStrides[DispatchDimCount - 3]);
    m_shaderUniforms.otherStrides[2] = static_cast<uint32_t>(dispatchOtherStrides[DispatchDimCount - 2]);
    m_shaderUniforms.otherStrides[3] = static_cast<uint32_t>(dispatchOtherStrides[DispatchDimCount - 1]);
    m_shaderUniforms.outputStrides[0] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 4]);
    m_shaderUniforms.outputStrides[1] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 3]);
    m_shaderUniforms.outputStrides[2] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 2]);
    m_shaderUniforms.outputStrides[3] = static_cast<uint32_t>(dispatchOutputStrides[DispatchDimCount - 1]);
    SetUniforms(m_shaderUniforms);

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
