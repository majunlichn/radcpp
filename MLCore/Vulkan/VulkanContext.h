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
    rad::Ref<VulkanTensorOpElementWiseUnary> m_opBitwiseNot;
    rad::Ref<VulkanTensorOpElementWiseUnary> m_opBitwiseAndScalar;
    rad::Ref<VulkanTensorOpElementWiseBinary> m_opBitwiseAnd;
    rad::Ref<VulkanTensorOpElementWiseUnary> m_opBitwiseOrScalar;
    rad::Ref<VulkanTensorOpElementWiseBinary> m_opBitwiseOr;
    rad::Ref<VulkanTensorOpElementWiseUnary> m_opBitwiseXorScalar;
    rad::Ref<VulkanTensorOpElementWiseBinary> m_opBitwiseXor;

    VulkanContext(rad::Ref<VulkanDevice> device);
    virtual ~VulkanContext() override;

    VulkanDevice* GetDevice();

    virtual void Fill(Tensor& input, const Scalar& value) override;
    virtual void Random(Tensor& input, const Scalar& from, const Scalar& to) override;

    virtual void Add(const Tensor& input, const Scalar& other, Tensor& output) override;
    virtual void Add(const Tensor& input, const Tensor& other, const Scalar& alpha, Tensor& output) override;

    virtual void Subtract(const Tensor& input, const Scalar& other, Tensor& output) override;
    virtual void Subtract(const Tensor& input, const Tensor& other, const Scalar& alpha, Tensor& output) override;

    virtual void Multiply(const Tensor& input, const Scalar& other, Tensor& output) override;
    virtual void Multiply(const Tensor& input, const Tensor& other, Tensor& output) override;

    virtual void Divide(const Tensor& input, const Scalar& other, Tensor& output) override;
    virtual void Divide(const Tensor& input, const Tensor& other, Tensor& output) override;

    virtual void Remainder(const Tensor& input, const Scalar& other, Tensor& output) override;
    virtual void Remainder(const Tensor& input, const Tensor& other, Tensor& output) override;

    virtual void BitwiseAnd(const Tensor& input, const Scalar& other, Tensor& output) override;
    virtual void BitwiseAnd(const Tensor& input, const Tensor& other, Tensor& output) override;

    virtual void BitwiseNot(const Tensor& input, Tensor& output) override;

    virtual void BitwiseOr(const Tensor& input, const Scalar& other, Tensor& output) override;
    virtual void BitwiseOr(const Tensor& input, const Tensor& other, Tensor& output) override;

    virtual void BitwiseXor(const Tensor& input, const Scalar& other, Tensor& output) override;
    virtual void BitwiseXor(const Tensor& input, const Tensor& other, Tensor& output) override;

}; // class VulkanContext

} // namespace ML
