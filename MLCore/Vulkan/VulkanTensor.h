#pragma once

#include <MLCore/Tensor.h>
#include <vkpp/Core/Common.h>

namespace ML
{

class VulkanDevice;
class VulkanContext;

class VulkanTensorStorage : public TensorStorage
{
public:
    // The base offset of the buffer range in bytes.
    VkDeviceSize m_bufferOffset = 0;
    // The buffer size required, must be rounded up to the nearest 4-byte boundary.
    VkDeviceSize m_bufferSize = 0;
    rad::Ref<vkpp::Buffer> m_buffer;

    VulkanTensorStorage(rad::Ref<VulkanDevice> device);
    ~VulkanTensorStorage();

    bool Init(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options = {});

    virtual void Read(void* data, size_t offset, size_t sizeInBytes) override;
    virtual void Write(const void* data, size_t offset, size_t sizeInBytes) override;

}; // class VulkanTensorStorage

} // namespace ML
