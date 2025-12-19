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
    rad::Ref<VulkanTensorOpElementWiseBinary> m_opAdd;

    VulkanContext(rad::Ref<VulkanDevice> device);
    virtual ~VulkanContext() override;

    VulkanDevice* GetDevice();
    vkpp::Device* GetDeviceImpl();

    virtual void FillConstant(Tensor* input, float value) override;
    virtual void FillConstant(Tensor* input, int value) override;

    virtual void Add(Tensor* input, Tensor* other, float alpha = 1.0f, Tensor* output = nullptr) override;
    virtual void Add(Tensor* input, Tensor* other, int alpha = 1, Tensor* output = nullptr) override;

}; // class VulkanContext

} // namespace ML
