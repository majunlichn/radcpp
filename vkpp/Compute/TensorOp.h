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

    void Execute(glm::uvec3 groupCount);

    rad::Ref<Device> m_device;
    rad::Ref<Buffer> m_uniformBuffer;
    vk::raii::DescriptorSetLayout m_descSetLayout = { nullptr };
    vk::raii::PipelineLayout m_pipelineLayout = { nullptr };
    rad::Ref<vkpp::Pipeline> m_pipeline;

    rad::Ref<DescriptorPool> m_descPool;
    vk::raii::DescriptorSets m_descSets = { nullptr };

    std::map<uint32_t, Tensor*> m_bindings;

    // Read-After-Write
    bool m_enable_PreExecute_MemoryBarrierRAW = false;
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

    struct Uniforms
    {
        glm::uvec4 sizes;
        glm::uvec4 inputStrides;
        glm::uvec4 outputStrides;
    };

    virtual bool Init(const TensorOpElementWiseUnaryDesc& desc);

    void UpdateUniforms();

    TensorOpElementWiseUnaryDesc m_desc = {};
    Uniforms m_uniforms = {};

}; // class TensorOpElementWise

} // namespace vkpp
