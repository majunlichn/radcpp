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

    rad::Ref<VulkanTensorOpForEach> m_opFill;
    rad::Ref<VulkanTensorOpElementWiseUnary> m_opAddScalar;
    rad::Ref<VulkanTensorOpElementWiseBinary> m_opAdd;
    rad::Ref<VulkanTensorOpElementWiseUnary> m_opSubtractScalar;
    rad::Ref<VulkanTensorOpElementWiseBinary> m_opSubtract;
    rad::Ref<VulkanTensorOpElementWiseUnary> m_opMultiplyScalar;
    rad::Ref<VulkanTensorOpElementWiseBinary> m_opMultiply;
    rad::Ref<VulkanTensorOpElementWiseUnary> m_opDivideScalar;
    rad::Ref<VulkanTensorOpElementWiseBinary> m_opDivide;
    rad::Ref<VulkanTensorOpElementWiseUnary> m_opRemainderScalar;
    rad::Ref<VulkanTensorOpElementWiseBinary> m_opRemainder;

    VulkanContext(rad::Ref<VulkanDevice> device);
    virtual ~VulkanContext() override;

    VulkanDevice* GetDevice();
    vkpp::Device* GetDeviceImpl();

    virtual void Fill(const Tensor& input, Scalar value) override;

    virtual void Add(const Tensor& input, const Scalar other, Tensor& output) override;
    virtual void Add(const Tensor& input, const Tensor& other, const Scalar alpha, Tensor& output) override;

    virtual void Subtract(const Tensor& input, const Scalar other, Tensor& output) override;
    virtual void Subtract(const Tensor& input, const Tensor& other, const Scalar alpha, Tensor& output) override;

    virtual void Multiply(const Tensor& input, const Scalar other, Tensor& output) override;
    virtual void Multiply(const Tensor& input, const Tensor& other, Tensor& output) override;

    virtual void Divide(const Tensor& input, const Scalar other, Tensor& output) override;
    virtual void Divide(const Tensor& input, const Tensor& other, Tensor& output) override;

    virtual void Remainder(const Tensor& input, const Scalar other, Tensor& output) override;
    virtual void Remainder(const Tensor& input, const Tensor& other, Tensor& output) override;

}; // class VulkanContext

} // namespace ML
