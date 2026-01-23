#pragma once

#include <MLCore/Vulkan/VulkanTensor.h>

#include <map>
#include <glm/glm.hpp>

namespace ML
{

class VulkanContext;

// Expected to be hold by VulkanContext.
class VulkanTensorOp : public rad::RefCounted<VulkanTensorOp>
{
public:
    VulkanContext* m_context;
    rad::Ref<vkpp::CommandStream> m_cmdStream;
    rad::Ref<vkpp::Buffer> m_uniformBuffer;
    rad::Ref<vkpp::DescriptorSetLayout> m_descSetLayout;
    rad::Ref<vkpp::PipelineLayout> m_pipelineLayout;

    rad::Ref<vkpp::DescriptorPool> m_descPool;
    rad::Ref<vkpp::DescriptorSet> m_descSet;

    std::map<uint32_t, const Tensor*> m_bindings;

    VulkanTensorOp(VulkanContext* context);
    virtual ~VulkanTensorOp();

    // binding[0]: the uniform buffer.
    // bindign[1~tensorCount]: the input/output tensors (storage buffers).
    void CreatePipelineLayout(size_t tensorCount,
        rad::ArrayRef<vk::PushConstantRange> pushConstantRanges = {});

    void SetUniforms(const void* data, size_t dataSize);
    template <rad::TriviallyCopyable T>
    void SetUniforms(const T& uniforms)
    {
        SetUniforms(&uniforms, sizeof(uniforms));
    }

    virtual void SetTensor(uint32_t binding, const Tensor& tensor);
    virtual void Execute() = 0;

    // Expand size dimensions (same element count and memory layout).
    static std::vector<size_t> ExpandTensorSizeND(rad::ArrayRef<size_t> sizes, size_t nd);
    // Expand stride dimensions (same element count and memory layout).
    static std::vector<size_t> ExpandTensorStrideND(rad::ArrayRef<size_t> strides, size_t nd);

    std::string GetShaderBinaryDir() const;

}; // class VulkanTensorOp

struct VulkanElementWiseShaderParams
{
    union
    {
        glm::f32vec4 f32;
        glm::i32vec4 i32;
        glm::u32vec4 u32;
        glm::f64vec4 f64;
        glm::i64vec4 i64;
        glm::u64vec4 u64;
        rad::Complex32 c32[4];
        rad::Complex64 c64[4];
        rad::Complex128 c128[4];
    } m_scalars;

    void SetScalars(DataType dataType, const Scalar& a, const Scalar& b = {}, const Scalar& c = {}, const Scalar& d = {});

}; // struct VulkanElementWiseShaderParams


class VulkanTensorOpForEach : public VulkanTensorOp
{
public:
    std::string m_opName;
    struct ShaderUniforms
    {
        glm::uvec4 sizes;
        glm::uvec4 strides;
        VulkanElementWiseShaderParams params;
    } m_shaderUniforms = {};

    static constexpr size_t DispatchDimCount = 4;
    std::map<DataType, rad::Ref<vkpp::Pipeline>> m_pipelines;

    VulkanTensorOpForEach(VulkanContext* context, std::string_view opName);
    ~VulkanTensorOpForEach();

    struct PushConstants
    {
        uint32_t inputIndexOffset;
    };

    const Tensor* GetInputTensor() { return m_bindings[1]; }

    virtual void Execute() override;

}; // VulkanTensorOpForEach

class VulkanTensorOpElementWiseUnary : public VulkanTensorOp
{
public:
    std::string m_opName;
    struct ShaderUniforms
    {
        glm::uvec4 sizes;
        glm::uvec4 inputStrides;
        glm::uvec4 outputStrides;
        VulkanElementWiseShaderParams params;
    } m_shaderUniforms = {};

    static constexpr size_t DispatchDimCount = 4;
    std::map<DataType, rad::Ref<vkpp::Pipeline>> m_pipelines;

    VulkanTensorOpElementWiseUnary(VulkanContext* context, std::string_view opName, rad::ArrayRef<DataType> dataTypes);
    ~VulkanTensorOpElementWiseUnary();

    struct PushConstants
    {
        uint32_t inputIndexOffset;
        uint32_t outputIndexOffset;
    };

    const Tensor* GetInputTensor() { return m_bindings[1]; }
    const Tensor* GetOutputTensor() { return m_bindings[2]; }

    virtual void Execute() override;

}; // VulkanTensorOpElementWiseUnary

class VulkanTensorOpElementWiseBinary : public VulkanTensorOp
{
public:
    std::string m_opName;
    struct ShaderUniforms
    {
        glm::uvec4 sizes;
        glm::uvec4 inputStrides;
        glm::uvec4 otherStrides;
        glm::uvec4 outputStrides;
        VulkanElementWiseShaderParams params;
    } m_shaderUniforms = {};

    static constexpr size_t DispatchDimCount = 4;
    std::map<DataType, rad::Ref<vkpp::Pipeline>> m_pipelines;

    VulkanTensorOpElementWiseBinary(VulkanContext* context, std::string_view opName, rad::ArrayRef<DataType> dataTypes);
    ~VulkanTensorOpElementWiseBinary();

    struct PushConstants
    {
        uint32_t inputIndexOffset;
        uint32_t otherIndexOffset;
        uint32_t outputIndexOffset;
    };

    const Tensor* GetInputTensor() { return m_bindings[1]; }
    const Tensor* GetOtherTensor() { return m_bindings[2]; }
    const Tensor* GetOutputTensor() { return m_bindings[3]; }

    virtual void Execute() override;

}; // VulkanTensorOpElementWiseBinary

} // namespace ML
