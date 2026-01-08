#include <MLCore/Vulkan/VulkanTensor.h>
#include <MLCore/Vulkan/VulkanDevice.h>
#include <MLCore/Vulkan/VulkanContext.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Buffer.h>

namespace ML
{

VulkanTensorStorage::VulkanTensorStorage(rad::Ref<VulkanDevice> device) :
    TensorStorage(std::move(device))
{
}

VulkanTensorStorage::~VulkanTensorStorage()
{
}

bool VulkanTensorStorage::Init(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options)
{
    m_dataType = dataType;
    m_sizes = sizes;
    const auto& strides = options.m_strides;
    if (strides.empty())
    {
        m_strides = MakeTensorStrides(sizes);
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

    VkDeviceSize bufferSize = VkDeviceSize(indexOfTheLastElement + 1) * GetElementSize(m_dataType);
    bufferSize = rad::Pow2AlignUp(bufferSize, VkDeviceSize(4));

    m_buffer = vkpp::Buffer::CreateStorage(static_cast<VulkanDevice*>(m_device.get())->m_impl, bufferSize);

    return true;
}

void VulkanTensorStorage::Read(void* data, size_t offset, size_t sizeInBytes)
{
    m_buffer->Read(data, offset, sizeInBytes);
}

void VulkanTensorStorage::Write(const void* data, size_t offset, size_t sizeInBytes)
{
    m_buffer->Write(data, offset, sizeInBytes);
}

} // namespace ML
