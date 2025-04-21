#include <vkpp/Core/Command.h>
#include <vkpp/Core/Device.h>


namespace vkpp
{

CommandPool::CommandPool(rad::Ref<Device> device, QueueFamily queueFamily, vk::CommandPoolCreateFlags flags) :
    m_device(std::move(device)),
    m_queueFamily(queueFamily)
{
    vk::CommandPoolCreateInfo createInfo(flags, m_device->GetQueueFamilyIndex(queueFamily));
    m_handle = m_device->m_handle.createCommandPool(createInfo);
}

CommandPool::~CommandPool()
{
}

vk::raii::CommandBuffers CommandPool::Allocate(vk::CommandBufferLevel level, uint32_t count)
{
    vk::CommandBufferAllocateInfo allocateInfo(m_handle, level, count);
    return vk::raii::CommandBuffers(m_device->m_handle, allocateInfo);
}

void CommandRecorder::SetMemoryBarrier(vk::PipelineStageFlags2KHR srcStageMask, vk::AccessFlags2KHR srcAccessMask, vk::PipelineStageFlags2KHR dstStageMask, vk::AccessFlags2KHR dstAccessMask)
{
    vk::MemoryBarrier2KHR barrier;
    barrier.srcStageMask = srcStageMask;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstStageMask = dstStageMask;
    barrier.dstAccessMask = dstAccessMask;
    vk::DependencyInfoKHR dependency;
    dependency.setMemoryBarriers(barrier);
    m_cmdBuffer.pipelineBarrier2(dependency);
}

void CommandRecorder::SetImageBarrier(vk::PipelineStageFlags2KHR srcStageMask, vk::AccessFlags2KHR srcAccessMask, vk::PipelineStageFlags2KHR dstStageMask, vk::AccessFlags2KHR dstAccessMask, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::Image image, const vk::ImageSubresourceRange& range)
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
    m_cmdBuffer.pipelineBarrier2(dependency);
}

void CommandRecorder::SetMemoryBarrier_ComputeToComputeRAW()
{
    vk::MemoryBarrier2 barrier;
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;

    vk::DependencyInfoKHR dependency;
    dependency.setMemoryBarriers(barrier);
    m_cmdBuffer.pipelineBarrier2(dependency);
}

void CommandRecorder::SetMemoryBarrier_ComputeToComputeWAR()
{
    // WAR hazards don't need availability or visibility operations between them -
    // execution dependencies are sufficient.
    // A pipeline barrier or event without a any access flags is an execution dependency.
    vk::MemoryBarrier2 barrier;
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;

    vk::DependencyInfoKHR dependency;
    dependency.setMemoryBarriers(barrier);
    m_cmdBuffer.pipelineBarrier2(dependency);
}

void CommandRecorder::SetMemoryBarrier_ComputeWriteToGraphicsIndexRead()
{
    vk::MemoryBarrier2 barrier;
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eIndexInput;
    barrier.dstAccessMask = vk::AccessFlagBits2::eIndexRead | vk::AccessFlagBits2::eMemoryRead;
    vk::DependencyInfoKHR dependency;
    dependency.setMemoryBarriers(barrier);
    m_cmdBuffer.pipelineBarrier2(dependency);
}

void CommandRecorder::SetMemoryBarrier_ComputeWriteToGraphicsIndirectCommandRead()
{
    vk::MemoryBarrier2 barrier;
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
    barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect;
    barrier.dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eMemoryRead;
    vk::DependencyInfoKHR dependency;
    dependency.setMemoryBarriers(barrier);
    m_cmdBuffer.pipelineBarrier2(dependency);
}

void CommandRecorder::SetImageBarrier_ComputeWriteToGraphicsSample(
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
    m_cmdBuffer.pipelineBarrier2(dependency);
}

void CommandRecorder::SetImageBarrier_ColorAttachmentToComputeSample(
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
    m_cmdBuffer.pipelineBarrier2(dependency);
}

void CommandRecorder::SetImageBarrier_DepthStencilAttachmentToComputeSample(
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
    m_cmdBuffer.pipelineBarrier2(dependency);
}

void CommandRecorder::SetImageBarrier_DepthStencilAttachmentToFragmentSample(
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
    m_cmdBuffer.pipelineBarrier2(dependency);
}

void CommandRecorder::SetImageBarrier_ColorAttachmentToFragmentSample(vk::Image image, const vk::ImageSubresourceRange& range)
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
    m_cmdBuffer.pipelineBarrier2(dependency);
}

void CommandRecorder::SetImageBarrier_FragmentSampleToColorAttachment(
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
    m_cmdBuffer.pipelineBarrier2(dependency);
}

void CommandRecorder::SetMemoryBarrier_ShaderWriteToHostRead(vk::PipelineStageFlagBits2 stage)
{
    vk::MemoryBarrier2 barrier;
    barrier.srcStageMask = stage;
    barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eHost;
    barrier.dstAccessMask = vk::AccessFlagBits2::eHostRead;

    vk::DependencyInfo dependency;
    dependency.setMemoryBarriers(barrier);
    m_cmdBuffer.pipelineBarrier2(dependency);
}

} // namespace vkpp
