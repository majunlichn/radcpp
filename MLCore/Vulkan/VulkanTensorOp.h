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

    std::map<uint32_t, VulkanTensor*> m_bindings;

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

    virtual void SetTensor(uint32_t binding, VulkanTensor* tensor);
    virtual void Execute() = 0;

    // Expand size dimensions (same element count and memory layout).
    static std::vector<size_t> ExpandTensorSizeND(rad::ArrayRef<size_t> sizes, size_t nd);
    // Expand stride dimensions (same element count and memory layout).
    static std::vector<size_t> ExpandTensorStrideND(rad::ArrayRef<size_t> strides, size_t nd);

}; // class VulkanTensorOp

class VulkanTensorOpForEach : public VulkanTensorOp
{
public:
    std::string m_opName;
    struct UniformData
    {
        glm::uvec4 sizes;
        glm::uvec4 strides;
        glm::uvec4 params;
    } m_uniformData = {};

    static constexpr size_t DispatchDimCount = 4;
    std::map<DataType, rad::Ref<vkpp::Pipeline>> m_pipelines;

    VulkanTensorOpForEach(VulkanContext* context, std::string_view opName);
    ~VulkanTensorOpForEach();

    struct PushConstants
    {
        uint32_t inputIndexOffset;
    };

    VulkanTensor* GetInputTensor() { return m_bindings[1]; }

    void SetParameters(glm::vec4 params)
    {
        std::memcpy(&m_uniformData.params, &params, sizeof(params));
    }

    void SetParameters(glm::ivec4 params)
    {
        std::memcpy(&m_uniformData.params, &params, sizeof(params));
    }

    virtual void Execute() override;

}; // VulkanTensorOpForEach

class VulkanTensorOpElementWiseUnary : public VulkanTensorOp
{
public:
    std::string m_opName;
    struct UniformData
    {
        glm::uvec4 sizes;
        glm::uvec4 inputStrides;
        glm::uvec4 outputStrides;
        glm::uvec4 params;
    } m_uniformData = {};

    static constexpr size_t DispatchDimCount = 4;
    std::map<DataType, rad::Ref<vkpp::Pipeline>> m_pipelines;

    VulkanTensorOpElementWiseUnary(VulkanContext* context, std::string_view opName);
    ~VulkanTensorOpElementWiseUnary();

    struct PushConstants
    {
        uint32_t inputIndexOffset;
        uint32_t outputIndexOffset;
    };

    VulkanTensor* GetInputTensor() { return m_bindings[1]; }
    VulkanTensor* GetOutputTensor() { return m_bindings[2]; }

    void SetParameters(glm::vec4 params)
    {
        std::memcpy(&m_uniformData.params, &params, sizeof(params));
    }

    void SetParameters(glm::ivec4 params)
    {
        std::memcpy(&m_uniformData.params, &params, sizeof(params));
    }

    virtual void Execute() override;

}; // VulkanTensorOpElementWiseUnary

class VulkanTensorOpElementWiseBinary : public VulkanTensorOp
{
public:
    std::string m_opName;
    struct UniformData
    {
        glm::uvec4 sizes;
        glm::uvec4 inputStrides;
        glm::uvec4 otherStrides;
        glm::uvec4 outputStrides;
        glm::uvec4 params;
    } m_uniformData = {};
    static constexpr size_t DispatchDimCount = 4;
    std::map<DataType, rad::Ref<vkpp::Pipeline>> m_pipelines;

    VulkanTensorOpElementWiseBinary(VulkanContext* context, std::string_view opName);
    ~VulkanTensorOpElementWiseBinary();

    void SetParameters(glm::vec4 params)
    {
        std::memcpy(&m_uniformData.params, &params, sizeof(params));
    }

    void SetParameters(glm::ivec4 params)
    {
        std::memcpy(&m_uniformData.params, &params, sizeof(params));
    }

    struct PushConstants
    {
        uint32_t inputIndexOffset;
        uint32_t otherIndexOffset;
        uint32_t outputIndexOffset;
    };

    VulkanTensor* GetInputTensor() { return m_bindings[1]; }
    VulkanTensor* GetOtherTensor() { return m_bindings[2]; }
    VulkanTensor* GetOutputTensor() { return m_bindings[3]; }

    virtual void Execute() override;

}; // VulkanTensorOpElementWiseBinary

} // namespace ML
