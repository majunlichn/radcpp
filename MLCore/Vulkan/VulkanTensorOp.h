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

struct ElementWiseParams
{
    union
    {
        glm::f32vec4 f32;
        glm::i32vec4 i32;
        glm::u32vec4 u32;
        glm::f64vec4 f64;
        glm::i64vec4 i64;
        glm::u64vec4 u64;
    } m_params;

    void Set(DataType dataType, const Scalar& a, const Scalar& b = {}, const Scalar& c = {}, const Scalar& d = {})
    {
        switch (dataType)
        {
        case ML::DataType::Float16:
        case ML::DataType::Float32:
            m_params.f32 = glm::f32vec4(
                static_cast<rad::Float32>(a), static_cast<rad::Float32>(b),
                static_cast<rad::Float32>(c), static_cast<rad::Float32>(d));
            break;
        case ML::DataType::Float64:
            m_params.f64 = glm::f64vec4(
                static_cast<rad::Float64>(a), static_cast<rad::Float64>(b),
                static_cast<rad::Float64>(c), static_cast<rad::Float64>(d));
            break;
        case ML::DataType::Sint8:
        case ML::DataType::Sint16:
        case ML::DataType::Sint32:
            m_params.i32 = glm::i32vec4(static_cast<int32_t>(a), static_cast<int32_t>(b), static_cast<int32_t>(c), static_cast<int32_t>(d));
            break;
        case ML::DataType::Sint64:
            m_params.i64 = glm::i64vec4(static_cast<int64_t>(a), static_cast<int64_t>(b), static_cast<int64_t>(c), static_cast<int64_t>(d));
            break;
        case ML::DataType::Uint8:
        case ML::DataType::Uint16:
        case ML::DataType::Uint32:
            m_params.u32 = glm::u32vec4(static_cast<uint32_t>(a), static_cast<uint32_t>(b), static_cast<uint32_t>(c), static_cast<uint32_t>(d));
            break;
        case ML::DataType::Uint64:
            m_params.u64 = glm::u64vec4(static_cast<uint64_t>(a), static_cast<uint64_t>(b), static_cast<uint64_t>(c), static_cast<uint64_t>(d));
            break;
        }
        RAD_UNREACHABLE();
    }

}; // struct ElementWiseShaderParams


class VulkanTensorOpForEach : public VulkanTensorOp
{
public:
    std::string m_opName;
    struct ShaderUniforms
    {
        glm::uvec4 sizes;
        glm::uvec4 strides;
        ElementWiseParams params;
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
        glm::uvec4 g_padding; // Unused, just for alignment
        ElementWiseParams params;
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
        ElementWiseParams params;
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
