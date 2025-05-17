#include <vkpp/Core/Command.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Buffer.h>
#include <vkpp/Core/Image.h>
#include <vkpp/Core/Pipeline.h>

namespace vkpp
{

CommandPool::CommandPool(rad::Ref<Device> device, QueueFamily queueFamily, vk::CommandPoolCreateFlags flags) :
    m_device(std::move(device)),
    m_queueFamily(queueFamily)
{
    vk::CommandPoolCreateInfo createInfo(flags, m_device->GetQueueFamilyIndex(queueFamily));
    m_wrapper = m_device->m_wrapper.createCommandPool(createInfo);
}

CommandPool::~CommandPool()
{
}

const DeviceDispatcher* CommandPool::GetDispatcher() const
{
    return m_device->GetDispatcher();
}

vk::raii::CommandBuffers CommandPool::Allocate(vk::CommandBufferLevel level, uint32_t count)
{
    vk::CommandBufferAllocateInfo allocateInfo(m_wrapper, level, count);
    return vk::raii::CommandBuffers(m_device->m_wrapper, allocateInfo);
}

CommandBuffer::CommandBuffer(rad::Ref<Device> device, vk::CommandPool poolHandle, vk::CommandBuffer bufferHandle) :
    m_device(std::move(device))
{
    m_wrapper = vk::raii::CommandBuffer(m_device->m_wrapper, bufferHandle, poolHandle);
}

CommandBuffer::CommandBuffer(
    rad::Ref<CommandPool> pool, vk::CommandBuffer bufferHandle) :
    m_device(pool->m_device),
    m_pool(std::move(pool))
{
    m_wrapper = vk::raii::CommandBuffer(m_device->m_wrapper, bufferHandle, pool->GetHandle());
}

CommandBuffer::~CommandBuffer()
{
    m_wrapper.clear();
    m_pool.reset();
    m_device.reset();
}

const DeviceDispatcher* CommandBuffer::GetDispatcher() const
{
    return m_device->GetDispatcher();
}

void CommandBuffer::Begin(vk::CommandBufferUsageFlags flags, vk::CommandBufferInheritanceInfo* pInheritanceInfo)
{
    vk::CommandBufferBeginInfo beginInfo = {};
    beginInfo.flags = flags;
    beginInfo.pInheritanceInfo = pInheritanceInfo;
    m_wrapper.begin(beginInfo);
}

void CommandBuffer::End()
{
    m_wrapper.end();
}

void CommandBuffer::BindPipeine(Pipeline* pipeline)
{
    m_wrapper.bindPipeline(pipeline->GetBindPoint(), pipeline->GetHandle());
}

void CommandBuffer::SetMemoryBarrier(vk::PipelineStageFlags2KHR srcStageMask, vk::AccessFlags2KHR srcAccessMask, vk::PipelineStageFlags2KHR dstStageMask, vk::AccessFlags2KHR dstAccessMask)
{
    vk::MemoryBarrier2KHR barrier;
    barrier.srcStageMask = srcStageMask;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstStageMask = dstStageMask;
    barrier.dstAccessMask = dstAccessMask;
    vk::DependencyInfoKHR dependency;
    dependency.setMemoryBarriers(barrier);
    m_wrapper.pipelineBarrier2(dependency);
}

void CommandBuffer::SetImageBarrier(vk::PipelineStageFlags2KHR srcStageMask, vk::AccessFlags2KHR srcAccessMask, vk::PipelineStageFlags2KHR dstStageMask, vk::AccessFlags2KHR dstAccessMask, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::Image image, const vk::ImageSubresourceRange& range)
{
    vk::ImageMemoryBarrier2KHR barrier;
    barrier.srcStageMask = srcStageMask;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstStageMask = dstStageMask;
    barrier.dstAccessMask = dstAccessMask;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.image = image;
    barrier.subresourceRange = range;
    vk::DependencyInfoKHR dependency;
    dependency.setImageMemoryBarriers(barrier);
    m_wrapper.pipelineBarrier2(dependency);
}

void CommandBuffer::SetMemoryBarrier_ComputeToComputeRAW()
{
    vk::MemoryBarrier2 barrier;
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;

    vk::DependencyInfoKHR dependency;
    dependency.setMemoryBarriers(barrier);
    m_wrapper.pipelineBarrier2(dependency);
}

void CommandBuffer::SetMemoryBarrier_ComputeToComputeWAR()
{
    // WAR hazards don't need availability or visibility operations between them -
    // execution dependencies are sufficient.
    // A pipeline barrier or event without a any access flags is an execution dependency.
    vk::MemoryBarrier2 barrier;
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;

    vk::DependencyInfoKHR dependency;
    dependency.setMemoryBarriers(barrier);
    m_wrapper.pipelineBarrier2(dependency);
}

void CommandBuffer::SetMemoryBarrier_ComputeWriteToGraphicsIndexRead()
{
    vk::MemoryBarrier2 barrier;
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eIndexInput;
    barrier.dstAccessMask = vk::AccessFlagBits2::eIndexRead | vk::AccessFlagBits2::eMemoryRead;
    vk::DependencyInfoKHR dependency;
    dependency.setMemoryBarriers(barrier);
    m_wrapper.pipelineBarrier2(dependency);
}

void CommandBuffer::SetMemoryBarrier_ComputeWriteToGraphicsIndirectCommandRead()
{
    vk::MemoryBarrier2 barrier;
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect;
    barrier.dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eMemoryRead;
    vk::DependencyInfoKHR dependency;
    dependency.setMemoryBarriers(barrier);
    m_wrapper.pipelineBarrier2(dependency);
}

void CommandBuffer::SetImageBarrier_ComputeWriteToGraphicsSample(
    vk::Image image, const vk::ImageSubresourceRange& range)
{
    vk::ImageMemoryBarrier2 barrier;
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
    barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
    barrier.oldLayout = vk::ImageLayout::eGeneral;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.image = image;
    barrier.subresourceRange = range;
    vk::DependencyInfoKHR dependency;
    dependency.setImageMemoryBarriers(barrier);
    m_wrapper.pipelineBarrier2(dependency);
}

void CommandBuffer::SetImageBarrier_ColorAttachmentToComputeSample(
    vk::Image image, const vk::ImageSubresourceRange& range)
{
    vk::ImageMemoryBarrier2 barrier;
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    barrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
    barrier.oldLayout = vk::ImageLayout::eAttachmentOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.image = image;
    barrier.subresourceRange = range;
    vk::DependencyInfoKHR dependency;
    dependency.setImageMemoryBarriers(barrier);
    m_wrapper.pipelineBarrier2(dependency);
}

void CommandBuffer::SetImageBarrier_DepthStencilAttachmentToComputeSample(
    vk::Image image, const vk::ImageSubresourceRange& range)
{
    vk::ImageMemoryBarrier2 barrier;
    barrier.srcStageMask =
        vk::PipelineStageFlagBits2::eEarlyFragmentTests |
        vk::PipelineStageFlagBits2::eLateFragmentTests;
    barrier.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
    barrier.oldLayout = vk::ImageLayout::eAttachmentOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.image = image;
    barrier.subresourceRange = range;
    vk::DependencyInfoKHR dependency;
    dependency.setImageMemoryBarriers(barrier);
    m_wrapper.pipelineBarrier2(dependency);
}

void CommandBuffer::SetImageBarrier_DepthStencilAttachmentToFragmentSample(
    vk::Image image, const vk::ImageSubresourceRange& range)
{
    vk::ImageMemoryBarrier2 barrier;
    barrier.srcStageMask =
        vk::PipelineStageFlagBits2::eEarlyFragmentTests |
        vk::PipelineStageFlagBits2::eLateFragmentTests;
    barrier.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
    barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
    barrier.oldLayout = vk::ImageLayout::eAttachmentOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.image = image;
    barrier.subresourceRange = range;
    vk::DependencyInfoKHR dependency;
    dependency.setImageMemoryBarriers(barrier);
    m_wrapper.pipelineBarrier2(dependency);
}

void CommandBuffer::SetImageBarrier_ColorAttachmentToFragmentSample(vk::Image image, const vk::ImageSubresourceRange& range)
{
    vk::ImageMemoryBarrier2 barrier;
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    barrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
    barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
    barrier.oldLayout = vk::ImageLayout::eAttachmentOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.image = image;
    barrier.subresourceRange = range;
    vk::DependencyInfoKHR dependency;
    dependency.setImageMemoryBarriers(barrier);
    m_wrapper.pipelineBarrier2(dependency);
}

void CommandBuffer::SetImageBarrier_FragmentSampleToColorAttachment(
    vk::Image image, const vk::ImageSubresourceRange& range)
{
    vk::ImageMemoryBarrier2 barrier;
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    barrier.oldLayout = vk::ImageLayout::eReadOnlyOptimal;
    barrier.newLayout = vk::ImageLayout::eAttachmentOptimal;
    barrier.image = image;
    barrier.subresourceRange = range;
    vk::DependencyInfoKHR dependency;
    dependency.setImageMemoryBarriers(barrier);
    m_wrapper.pipelineBarrier2(dependency);
}

void CommandBuffer::SetMemoryBarrier_ShaderWriteToHostRead(vk::PipelineStageFlagBits2 stage)
{
    vk::MemoryBarrier2 barrier;
    barrier.srcStageMask = stage;
    barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eHost;
    barrier.dstAccessMask = vk::AccessFlagBits2::eHostRead;

    vk::DependencyInfo dependency;
    dependency.setMemoryBarriers(barrier);
    m_wrapper.pipelineBarrier2(dependency);
}

} // namespace vkpp
