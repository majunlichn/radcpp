#pragma once

#include <radcpp/GPU/VulkanCommon.h>

namespace rad
{

class VulkanDevice;
class VulkanBufferView;

class VulkanBuffer : public RefCounted<VulkanBuffer>
{
public:
    static Ref<VulkanBuffer> Create(Ref<VulkanDevice> device,
        vk::DeviceSize size, vk::BufferUsageFlags usage,
        VmaMemoryUsage memoryUsage = VMA_MEMORY_USAGE_AUTO,
        VmaAllocationCreateFlags allocFlags = 0);

    // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
    static Ref<VulkanBuffer> CreateUniform(Ref<VulkanDevice> device, vk::DeviceSize size)
    {
        return Create(device, size, vk::BufferUsageFlagBits::eUniformBuffer,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }
    static Ref<VulkanBuffer> CreateStagingUpload(Ref<VulkanDevice> device, vk::DeviceSize size)
    {
        return Create(device, size, vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }
    static Ref<VulkanBuffer> CreateStagingReadback(Ref<VulkanDevice> device, vk::DeviceSize size)
    {
        return Create(device, size, vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }

    Ref<VulkanDevice> m_device;
    vk::Buffer m_handle = nullptr;
    vk::DeviceSize m_size = 0;
    vk::BufferUsageFlags m_usage;
    VmaAllocation m_alloc = nullptr;
    VmaAllocationInfo m_allocInfo = {};
    vk::MemoryPropertyFlags m_memPropFlags;

    VulkanBuffer(Ref<VulkanDevice> device,
        const vk::BufferCreateInfo& bufferInfo, const VmaAllocationCreateInfo& allocCreateInfo);
    ~VulkanBuffer();

    vk::Buffer GetHandle() const { return m_handle; }

    bool IsHostVisible() const { return (m_memPropFlags & vk::MemoryPropertyFlagBits::eHostVisible).m_mask; }
    bool IsHostCoherent() const { return (m_memPropFlags & vk::MemoryPropertyFlagBits::eHostCoherent).m_mask; }

    void* MapMemory();
    void UnmapMemory();

    Ref<VulkanBufferView> CreateView(
        vk::Format format,
        vk::DeviceSize offset = 0,
        vk::DeviceSize range = vk::WholeSize,
        vk::BufferViewCreateFlags flags = {});

    void Read(void* data, vk::DeviceSize offset, vk::DeviceSize dataSize);
    void Write(const void* data, vk::DeviceSize offset, vk::DeviceSize dataSize);

}; // class VulkanBuffer

class VulkanBufferView : public RefCounted<VulkanBufferView>
{
public:
    VulkanBufferView(Ref<VulkanBuffer> buffer, const vk::BufferViewCreateInfo& createInfo);
    ~VulkanBufferView();

    Ref<VulkanBuffer> m_buffer;
    vk::raii::BufferView m_handle = { nullptr };
    vk::Format m_format = vk::Format::eUndefined;
    vk::DeviceSize m_offset = 0;
    vk::DeviceSize m_range = 0;

}; // class VulkanBufferView

} // namespace rad
