#pragma once

#include <vkpp/Compute/Tensor.h>
#include <vkpp/Core/Descriptor.h>
#include <vkpp/Core/Pipeline.h>

#include <glm/glm.hpp>

namespace vkpp
{

class TensorOp : public rad::RefCounted<TensorOp>
{
public:
    TensorOp(rad::Ref<Device> device);
    virtual ~TensorOp();

    static const char* GetDataTypeShaderString(vk::ComponentTypeKHR dataType);

    // binding[0]: the uniform buffer.
    // bindign[1~tensorCount]: the input/output tensors (storage buffers).
    void CreatePipelineLayouts(size_t tensorCount,
        rad::ArrayRef<vk::PushConstantRange> pushConstantRanges = {});

    void SetUniforms(const void* data, size_t dataSize);
    template <rad::TriviallyCopyable T>
    void SetUniforms(const T& uniforms)
    {
        SetUniforms(&uniforms, sizeof(uniforms));
    }

    void SetTensor(uint32_t binding, Tensor* tensors);
    virtual void Execute() = 0;

    rad::Ref<Device> m_device;
    rad::Ref<CommandStream> m_cmdStream;
    rad::Ref<Buffer> m_uniformBuffer;
    rad::Ref<DescriptorSetLayout> m_descSetLayout;
    rad::Ref<PipelineLayout> m_pipelineLayout;
    rad::Ref<vkpp::Pipeline> m_pipeline;

    rad::Ref<DescriptorPool> m_descPool;
    rad::Ref<DescriptorSet> m_descSet;

    std::map<uint32_t, Tensor*> m_bindings;

    std::vector<vk::MemoryBarrier2> m_memoryBarriers;

    std::vector<SubmitWaitInfo> m_executeWaits;
    std::vector<vk::Semaphore> m_executeSignalSemaphores;

}; // class TensorOp

// binding[1]: the input tensor.
// binding[2]: the output tensor.
struct TensorOpElementWiseUnaryDesc
{
    std::string opName;
    vk::ComponentTypeKHR dataType;
    std::vector<size_t> sizes;
    std::vector<size_t> inputStrides;
    std::vector<size_t> outputStrides;
};

class TensorOpElementWiseUnary : public TensorOp
{
public:
    TensorOpElementWiseUnary(rad::Ref<Device> device);
    ~TensorOpElementWiseUnary();

    struct PushConstants
    {
        uint32_t inputIndexOffset;
        uint32_t outputIndexOffset;
    };

    struct Uniforms
    {
        glm::uvec4 sizes;
        glm::uvec4 inputStrides;
        glm::uvec4 outputStrides;
    };

    virtual bool Init(const TensorOpElementWiseUnaryDesc& desc);

    void UpdateUniforms();

    virtual void Execute() override;

    TensorOpElementWiseUnaryDesc m_desc = {};
    Uniforms m_uniforms = {};

}; // class TensorOpElementWise

} // namespace vkpp
