#pragma once

#include <MLCore/Context.h>
#include <MLCore/Vulkan/VulkanTensorOp.h>
#include <vkpp/Core/Common.h>

namespace ML
{

class VulkanDevice;

class VulkanContext : public Context
{
public:
    std::vector<vk::MemoryBarrier2> m_memoryBarriers;

    std::vector<vkpp::SubmitWaitInfo> m_submitWaits;
    std::vector<vk::Semaphore> m_submitSignals;

    rad::Ref<VulkanTensorOpForEach> m_opFillConstant;
    rad::Ref<VulkanTensorOpElementWiseUnary> m_opAddScalar;
    rad::Ref<VulkanTensorOpElementWiseBinary> m_opAdd;
    rad::Ref<VulkanTensorOpElementWiseUnary> m_opSubtractScalar;
    rad::Ref<VulkanTensorOpElementWiseBinary> m_opSubtract;
    rad::Ref<VulkanTensorOpElementWiseUnary> m_opMultiplyScalar;
    rad::Ref<VulkanTensorOpElementWiseBinary> m_opMultiply;
    rad::Ref<VulkanTensorOpElementWiseUnary> m_opDivideScalar;
    rad::Ref<VulkanTensorOpElementWiseBinary> m_opDivide;

    VulkanContext(rad::Ref<VulkanDevice> device);
    virtual ~VulkanContext() override;

    VulkanDevice* GetDevice();
    vkpp::Device* GetDeviceImpl();

    virtual void FillConstant(Tensor* input, float value) override;
    virtual void FillConstant(Tensor* input, int value) override;

    virtual void AddScalar(Tensor* input, float other, Tensor* output = nullptr) override;
    virtual void AddScalar(Tensor* input, int other, Tensor* output = nullptr) override;

    virtual void Add(Tensor* input, Tensor* other, float alpha = 1.0f, Tensor* output = nullptr) override;
    virtual void Add(Tensor* input, Tensor* other, int alpha = 1, Tensor* output = nullptr) override;

    virtual void SubtractScalar(Tensor* input, float other, Tensor* output = nullptr) override;
    virtual void SubtractScalar(Tensor* input, int other, Tensor* output = nullptr) override;

    virtual void Subtract(Tensor* input, Tensor* other, float alpha = 1.0f, Tensor* output = nullptr) override;
    virtual void Subtract(Tensor* input, Tensor* other, int alpha = 1, Tensor* output = nullptr) override;

    virtual void MultiplyScalar(Tensor* input, float other, Tensor* output = nullptr) override;
    virtual void MultiplyScalar(Tensor* input, int other, Tensor* output = nullptr) override;
    virtual void Multiply(Tensor* input, Tensor* other, Tensor* output = nullptr) override;

    virtual void DivideScalar(Tensor* input, float other, Tensor* output = nullptr) override;
    virtual void DivideScalar(Tensor* input, int other, Tensor* output = nullptr) override;

    virtual void Divide(Tensor* input, Tensor* other, Tensor* output = nullptr) override;

}; // class VulkanContext

} // namespace ML
