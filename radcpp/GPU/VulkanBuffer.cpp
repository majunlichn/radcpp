#include <radcpp/GPU/VulkanBuffer.h>
#include <radcpp/GPU/VulkanDevice.h>

namespace rad
{

Ref<VulkanBuffer> VulkanBuffer::Create(Ref<VulkanDevice> device,
    vk::DeviceSize size, vk::BufferUsageFlags usage,
    VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags allocFlags)
{
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = memoryUsage;
    allocInfo.flags = allocFlags;
    return RAD_NEW VulkanBuffer(std::move(device), bufferInfo, allocInfo);
}

VulkanBuffer::VulkanBuffer(Ref<VulkanDevice> device,
    const vk::BufferCreateInfo& bufferInfo, const VmaAllocationCreateInfo& allocCreateInfo) :
    m_device(std::move(device))
{
    static_assert(sizeof(vk::Buffer) == sizeof(VkBuffer));
    static_assert(sizeof(vk::BufferCreateInfo) == sizeof(VkBufferCreateInfo));
    VK_CHECK(vmaCreateBuffer(m_device->m_allocator,
        reinterpret_cast<const VkBufferCreateInfo*>(&bufferInfo),
        &allocCreateInfo,
        reinterpret_cast<VkBuffer*>(&m_handle),
        &m_alloc, &m_allocInfo));
    if (m_handle && m_alloc)
    {
        m_size = bufferInfo.size;
        m_usage = bufferInfo.usage;
        vmaGetAllocationMemoryProperties(m_device->m_allocator, m_alloc,
            reinterpret_cast<VkMemoryPropertyFlags*>(&m_memPropFlags));
    }
}

VulkanBuffer::~VulkanBuffer()
{
    if (m_handle && m_alloc)
    {
        vmaDestroyBuffer(m_device->m_allocator, m_handle, m_alloc);
        m_handle = nullptr;
        m_alloc = nullptr;
    }
}

void* VulkanBuffer::MapMemory()
{
    void* pMappedData = nullptr;
    VK_CHECK(vmaMapMemory(m_device->m_allocator, m_alloc, &pMappedData));
    return pMappedData;
}

void VulkanBuffer::UnmapMemory()
{
    vmaUnmapMemory(m_device->m_allocator, m_alloc);
}

Ref<VulkanBufferView> VulkanBuffer::CreateView(
    vk::Format format, vk::DeviceSize offset, vk::DeviceSize range, vk::BufferViewCreateFlags flags)
{
    vk::BufferViewCreateInfo viewInfo;
    viewInfo.buffer = m_handle;
    viewInfo.format = format;
    viewInfo.offset = offset;
    viewInfo.range = range;
    return RAD_NEW VulkanBufferView(this, viewInfo);
}

void VulkanBuffer::Read(void* data, vk::DeviceSize offset, vk::DeviceSize dataSize)
{
    assert(m_memPropFlags & vk::MemoryPropertyFlagBits::eHostVisible);
    if (m_allocInfo.pMappedData)
    {
        if (!(m_memPropFlags & vk::MemoryPropertyFlagBits::eHostCoherent))
        {
            vmaFlushAllocation(m_device->m_allocator, m_alloc, offset, dataSize);
        }
        memcpy(data, static_cast<char*>(m_allocInfo.pMappedData) + offset, dataSize);
    }
    else
    {
        void* mapped = MapMemory();
        if (mapped)
        {
            if (!(m_memPropFlags & vk::MemoryPropertyFlagBits::eHostCoherent))
            {
                vmaFlushAllocation(m_device->m_allocator, m_alloc, offset, dataSize);
            }
            memcpy(data, mapped, dataSize);
            UnmapMemory();
        }
    }
}

void VulkanBuffer::Write(const void* data, vk::DeviceSize offset, vk::DeviceSize dataSize)
{
    assert(m_memPropFlags & vk::MemoryPropertyFlagBits::eHostVisible);
    if (m_allocInfo.pMappedData)
    {
        memcpy(static_cast<char*>(m_allocInfo.pMappedData) + offset, data, dataSize);
        if (!(m_memPropFlags & vk::MemoryPropertyFlagBits::eHostCoherent))
        {
            vmaFlushAllocation(m_device->m_allocator, m_alloc, offset, dataSize);
        }
    }
    else
    {
        void* mapped = MapMemory();
        if (mapped)
        {
            memcpy(mapped, data, dataSize);
            if (!(m_memPropFlags & vk::MemoryPropertyFlagBits::eHostCoherent))
            {
                vmaFlushAllocation(m_device->m_allocator, m_alloc, offset, dataSize);
            }
            UnmapMemory();
        }
    }
}

VulkanBufferView::VulkanBufferView(
    Ref<VulkanBuffer> buffer, const vk::BufferViewCreateInfo& createInfo) :
    m_buffer(std::move(buffer))
{
    static_assert(sizeof(vk::BufferView) == sizeof(VkBufferView));
    static_assert(sizeof(vk::BufferViewCreateInfo) == sizeof(VkBufferViewCreateInfo));
    m_handle = m_buffer->m_device->m_handle.createBufferView(createInfo);
    if (static_cast<vk::BufferView>(m_handle))
    {
        m_format = createInfo.format;
        m_offset = createInfo.offset;
        m_range = createInfo.range;
    }
}

VulkanBufferView::~VulkanBufferView()
{
}

} // namespace rad
