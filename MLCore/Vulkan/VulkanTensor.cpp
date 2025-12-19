#include <MLCore/Vulkan/VulkanTensor.h>
#include <MLCore/Vulkan/VulkanDevice.h>
#include <MLCore/Vulkan/VulkanContext.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Buffer.h>

namespace ML
{

VulkanTensor::VulkanTensor(rad::Ref<VulkanDevice> device) :
    Tensor(std::move(device))
{
}

VulkanTensor::~VulkanTensor()
{
}

VulkanDevice* VulkanTensor::GetDevice()
{
    return static_cast<VulkanDevice*>(m_device.get());
}

vkpp::Device* VulkanTensor::GetDeviceImpl()
{
    return static_cast<VulkanDevice*>(m_device.get())->m_impl.get();
}

bool VulkanTensor::Init(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options)
{
    m_dataType = dataType;
    m_sizes = sizes;
    const auto& strides = options.m_strides;
    if (strides.empty())
    {
        m_strides = MakeStrides(sizes);
    }
    else
    {
        assert(strides.size() == sizes.size());
        m_strides = strides;
    }

    size_t indexOfTheLastElement = 0;
    for (size_t i = 0; i < m_sizes.size(); ++i)
    {
        indexOfTheLastElement += (m_sizes[i] - 1) * m_strides[i];
    }
    size_t elementCount = GetElementCount();
    if (indexOfTheLastElement + 1 == elementCount)
    {
        m_isContiguous = true;
    }

    m_bufferSize = VkDeviceSize(indexOfTheLastElement + 1) * GetElementSize(m_dataType);
    m_bufferSize = rad::Pow2AlignUp(m_bufferSize, VkDeviceSize(4));

    m_buffer = vkpp::Buffer::CreateStorage(GetDeviceImpl(), m_bufferSize);

    return true;
}

void VulkanTensor::Read(void* data, size_t offset, size_t sizeInBytes)
{
    m_buffer->Read(data, offset, sizeInBytes);
}

void VulkanTensor::Write(const void* data, size_t offset, size_t sizeInBytes)
{
    m_buffer->Write(data, offset, sizeInBytes);
}

} // namespace ML
