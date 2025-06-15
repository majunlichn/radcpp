#include <vkpp/Core/Buffer.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Command.h>

namespace vkpp
{

rad::Ref<Buffer> Buffer::Create(
    rad::Ref<Device> device,
    vk::DeviceSize size, vk::BufferUsageFlags usage,
    VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags allocFlags)
{
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = memoryUsage;
    allocInfo.flags = allocFlags;
    return RAD_NEW Buffer(std::move(device), bufferInfo, allocInfo);
}

Buffer::Buffer(rad::Ref<Device> device,
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

    m_cmdPool = m_device->CreateCommandPool(QueueFamily::Universal, vk::CommandPoolCreateFlagBits::eTransient);
}

Buffer::~Buffer()
{
    if (m_handle && m_alloc)
    {
        vmaDestroyBuffer(m_device->m_allocator, m_handle, m_alloc);
        m_handle = nullptr;
        m_alloc = nullptr;
    }
}

void* Buffer::MapMemory()
{
    void* pMappedData = nullptr;
    VK_CHECK(vmaMapMemory(m_device->m_allocator, m_alloc, &pMappedData));
    return pMappedData;
}

void Buffer::UnmapMemory()
{
    vmaUnmapMemory(m_device->m_allocator, m_alloc);
}

rad::Ref<BufferView> Buffer::CreateView(
    vk::Format format, vk::DeviceSize offset, vk::DeviceSize range, vk::BufferViewCreateFlags flags)
{
    vk::BufferViewCreateInfo viewInfo;
    viewInfo.buffer = m_handle;
    viewInfo.format = format;
    viewInfo.offset = offset;
    viewInfo.range = range;
    return RAD_NEW BufferView(this, viewInfo);
}

void Buffer::ReadHost(void* data, vk::DeviceSize offset, vk::DeviceSize dataSize)
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
        void* pMappedData = MapMemory();
        if (pMappedData)
        {
            if (!(m_memPropFlags & vk::MemoryPropertyFlagBits::eHostCoherent))
            {
                vmaFlushAllocation(m_device->m_allocator, m_alloc, offset, dataSize);
            }
            memcpy(data, pMappedData, dataSize);
            UnmapMemory();
        }
    }
}

void Buffer::WriteHost(const void* data, vk::DeviceSize offset, vk::DeviceSize dataSize)
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
        void* pMappedData = MapMemory();
        if (pMappedData)
        {
            memcpy(pMappedData, data, dataSize);
            if (!(m_memPropFlags & vk::MemoryPropertyFlagBits::eHostCoherent))
            {
                vmaFlushAllocation(m_device->m_allocator, m_alloc, offset, dataSize);
            }
            UnmapMemory();
        }
    }
}

void Buffer::Read(void* data, vk::DeviceSize offset, vk::DeviceSize dataSize)
{
    if (IsHostVisible())
    {
        Read(data, offset, dataSize);
    }
    else
    {
        rad::Ref<Buffer> stagingBuffer = CreateStagingReadback(m_device, dataSize);

        rad::Ref<CommandBuffer> cmdBuffer = m_cmdPool->AllocatePrimary();
        cmdBuffer->Begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        // Generally considered more efficient to do a global memory barrier than per-resource barriers?
        vk::MemoryBarrier2 deviceRAW;
        deviceRAW.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
        deviceRAW.srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eMemoryWrite;
        deviceRAW.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
        deviceRAW.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
        vk::DependencyInfo dependency;
        dependency.setMemoryBarriers(deviceRAW);
        cmdBuffer->SetPipelineBarrier2(dependency);
        vk::BufferCopy copyRegion = {};
        copyRegion.srcOffset = offset;
        copyRegion.dstOffset = 0;
        copyRegion.size = dataSize;
        cmdBuffer->CopyBuffer(GetHandle(), stagingBuffer->GetHandle(), copyRegion);
        vk::MemoryBarrier2 stagingRAW;
        stagingRAW.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        stagingRAW.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        stagingRAW.dstStageMask = vk::PipelineStageFlagBits2::eHost;
        stagingRAW.dstAccessMask = vk::AccessFlagBits2::eHostRead;
        dependency.setMemoryBarriers(stagingRAW);
        cmdBuffer->SetPipelineBarrier2(dependency);
        cmdBuffer->End();

        m_device->GetQueue(vkpp::QueueFamily::Universal)->
            ExecuteSync(cmdBuffer->GetHandle(), {}, {});
        stagingBuffer->ReadHost(data, 0, dataSize);
    }
}

void Buffer::Write(const void* data, vk::DeviceSize offset, vk::DeviceSize dataSize)
{
    if (IsHostVisible())
    {
        WriteHost(data, offset, dataSize);
    }
    else
    {
        rad::Ref<Buffer> stagingBuffer = CreateStagingUpload(m_device, dataSize);
        stagingBuffer->WriteHost(data, 0, dataSize);
        rad::Ref<CommandBuffer> cmdBuffer = m_cmdPool->AllocatePrimary();
        cmdBuffer->Begin(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        vk::BufferCopy copyRegion = {};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = offset;
        copyRegion.size = dataSize;
        cmdBuffer->CopyBuffer(stagingBuffer->GetHandle(), GetHandle(), copyRegion);
        vk::MemoryBarrier2 deviceRAW;
        deviceRAW.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        deviceRAW.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        deviceRAW.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
        deviceRAW.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eMemoryRead;
        vk::DependencyInfo dependency;
        dependency.setMemoryBarriers(deviceRAW);
        cmdBuffer->SetPipelineBarrier2(dependency);
        cmdBuffer->End();

        m_device->GetQueue(vkpp::QueueFamily::Universal)->
            ExecuteSync(cmdBuffer->GetHandle(), {}, {});
    }
}

BufferView::BufferView(
    rad::Ref<Buffer> buffer, const vk::BufferViewCreateInfo& createInfo) :
    m_buffer(std::move(buffer))
{
    static_assert(sizeof(vk::BufferView) == sizeof(VkBufferView));
    static_assert(sizeof(vk::BufferViewCreateInfo) == sizeof(VkBufferViewCreateInfo));
    m_wrapper = m_buffer->m_device->m_wrapper.createBufferView(createInfo);
    if (static_cast<vk::BufferView>(m_wrapper))
    {
        m_format = createInfo.format;
        m_offset = createInfo.offset;
        m_range = createInfo.range;
    }
}

BufferView::~BufferView()
{
}

} // namespace vkpp
