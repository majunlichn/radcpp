#pragma once

#include <MLCore/Device.h>
#include <vkpp/Core/Instance.h>
#include <vkpp/Core/Device.h>

namespace ML
{

class Context;
class Tensor;

// Backend: Vulkan
class VulkanDevice : public Device
{
public:
    rad::Ref<vkpp::Device> m_impl;

    VulkanDevice(rad::Ref<vkpp::Device> impl);
    ~VulkanDevice();

    virtual rad::Ref<Context> CreateContext() override;
    virtual rad::Ref<TensorStorage> CreateTensorStorage(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options = {}) override;

    virtual bool IsDataTypeSupported(DataType dataType) const override;
    virtual bool IsDataTypeComputable(DataType dataType) const override;

}; // class VulkanDevice

} // namespace ML
