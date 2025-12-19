#pragma once

#include <MLCore/Tensor.h>
#include <vkpp/Core/Common.h>

namespace ML
{

class VulkanDevice;
class VulkanContext;

class VulkanTensor : public Tensor
{
public:
    bool m_isContiguous = false;
    // The base offset of the buffer range in bytes.
    VkDeviceSize m_bufferOffset = 0;
    // The buffer size required, must be rounded up to the nearest 4-byte boundary.
    VkDeviceSize m_bufferSize = 0;
    rad::Ref<vkpp::Buffer> m_buffer;

    VulkanTensor(rad::Ref<VulkanDevice> device);
    ~VulkanTensor();

    VulkanDevice* GetDevice();
    vkpp::Device* GetDeviceImpl();

    bool Init(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options = {});

    virtual void* GetData() override { return nullptr; }

    virtual void Read(void* data, size_t offset, size_t sizeInBytes) override;
    virtual void Write(const void* data, size_t offset, size_t sizeInBytes) override;

}; // class VulkanTensor

} // namespace ML
