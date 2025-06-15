#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Buffer : public rad::RefCounted<Buffer>
{
public:
    static rad::Ref<Buffer> Create(
        rad::Ref<Device> device,
        vk::DeviceSize size, vk::BufferUsageFlags usage,
        VmaMemoryUsage memoryUsage = VMA_MEMORY_USAGE_AUTO,
        VmaAllocationCreateFlags allocFlags = 0);

    // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
    static rad::Ref<Buffer> CreateUniform(rad::Ref<Device> device, vk::DeviceSize size)
    {
        return Create(device, size, vk::BufferUsageFlagBits::eUniformBuffer,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }

    static rad::Ref<Buffer> CreateStagingUpload(rad::Ref<Device> device, vk::DeviceSize size)
    {
        return Create(device, size, vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }

    static rad::Ref<Buffer> CreateStagingReadback(rad::Ref<Device> device, vk::DeviceSize size)
    {
        return Create(device, size, vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }

    static rad::Ref<Buffer> CreateStorage(rad::Ref<Device> device, vk::DeviceSize size)
    {
        return Create(device, size, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
            VMA_MEMORY_USAGE_AUTO);
    }

    rad::Ref<Device> m_device;
    vk::Buffer m_handle = nullptr;
    vk::DeviceSize m_size = 0;
    vk::BufferUsageFlags m_usage;
    VmaAllocation m_alloc = nullptr;
    VmaAllocationInfo m_allocInfo = {};
    vk::MemoryPropertyFlags m_memPropFlags;

    rad::Ref<CommandPool> m_cmdPool;

    Buffer(rad::Ref<Device> device,
        const vk::BufferCreateInfo& bufferInfo, const VmaAllocationCreateInfo& allocCreateInfo);
    ~Buffer();

    vk::Buffer GetHandle() const { return m_handle; }
    vk::DeviceSize GetSize() const { return m_size; }

    bool IsHostVisible() const { return (m_memPropFlags & vk::MemoryPropertyFlagBits::eHostVisible).m_mask; }
    bool IsHostCoherent() const { return (m_memPropFlags & vk::MemoryPropertyFlagBits::eHostCoherent).m_mask; }

    void* MapMemory();
    void UnmapMemory();

    rad::Ref<BufferView> CreateView(
        vk::Format format,
        vk::DeviceSize offset = 0,
        vk::DeviceSize range = vk::WholeSize,
        vk::BufferViewCreateFlags flags = {});

    void ReadHost(void* data, vk::DeviceSize offset, vk::DeviceSize dataSize);
    void ReadHost(void* data) { ReadHost(data, 0, GetSize()); };
    void WriteHost(const void* data, vk::DeviceSize offset, vk::DeviceSize dataSize);
    void WriteHost(const void* data) { WriteHost(data, 0, GetSize()); }

    void Read(void* data, vk::DeviceSize offset, vk::DeviceSize dataSize);
    void Read(void* data) { Read(data, 0, GetSize()); }
    void Write(const void* data, vk::DeviceSize offset, vk::DeviceSize dataSize);
    void Write(const void* data) { Write(data, 0, GetSize()); }

}; // class Buffer

class BufferView : public rad::RefCounted<BufferView>
{
public:
    BufferView(rad::Ref<Buffer> buffer, const vk::BufferViewCreateInfo& createInfo);
    ~BufferView();

    rad::Ref<Buffer> m_buffer;
    vk::raii::BufferView m_wrapper = { nullptr };
    vk::Format m_format = vk::Format::eUndefined;
    vk::DeviceSize m_offset = 0;
    vk::DeviceSize m_range = 0;

}; // class BufferView

} // namespace vkpp
