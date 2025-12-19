#include <MLCore/Vulkan/VulkanDevice.h>
#include <MLCore/Vulkan/VulkanContext.h>
#include <MLCore/Vulkan/VulkanTensor.h>
#include <vkpp/Core/Instance.h>

namespace ML
{

VulkanDevice::VulkanDevice(rad::Ref<vkpp::Device> impl) :
    m_impl(std::move(impl))
{
}

VulkanDevice::~VulkanDevice()
{
}

rad::Ref<Context> VulkanDevice::CreateContext()
{
    return RAD_NEW VulkanContext(this);
}

rad::Ref<Tensor> VulkanDevice::CreateTensor(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options)
{
    if (dataType == DataType::Unknown)
    {
        dataType = DataType::Float32;
    }
    rad::Ref<VulkanTensor> tensor = RAD_NEW VulkanTensor(this);
    if (tensor->Init(sizes, dataType, options))
    {
        return tensor;
    }
    else
    {
        return nullptr;
    }
}

bool VulkanDevice::IsDataTypeSupported(DataType dataType) const
{
    if ((dataType == DataType::Float16) ||
        (dataType == DataType::Float32) ||
        (dataType == DataType::Float64) ||
        (dataType == DataType::Sint8) ||
        (dataType == DataType::Sint16) ||
        (dataType == DataType::Sint32) ||
        (dataType == DataType::Sint64) ||
        (dataType == DataType::Uint8) ||
        (dataType == DataType::Uint16) ||
        (dataType == DataType::Uint32) ||
        (dataType == DataType::Uint64))
    {
        return true;
    }
    else if (dataType == DataType::BFloat16)
    {
        return m_impl->m_shaderBfloat16Features.shaderBFloat16Type;
    }
    else if (dataType == DataType::Float8E4M3)
    {
        return m_impl->m_shaderFloat8Features.shaderFloat8;
    }
    else if (dataType == DataType::Float8E5M2)
    {
        return m_impl->m_shaderFloat8Features.shaderFloat8;
    }
    return false;
}

bool VulkanDevice::IsDataTypeComputable(DataType dataType) const
{
    if ((dataType == DataType::Float16) ||
        (dataType == DataType::Float32) ||
        (dataType == DataType::Float64) ||
        (dataType == DataType::Sint8) ||
        (dataType == DataType::Sint16) ||
        (dataType == DataType::Sint32) ||
        (dataType == DataType::Sint64) ||
        (dataType == DataType::Uint8) ||
        (dataType == DataType::Uint16) ||
        (dataType == DataType::Uint32) ||
        (dataType == DataType::Uint64))
    {
        return true;
    }
    else
    {
        return false;
    }
}

} // namespace ML
