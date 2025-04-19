#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Device;

class CommandPool : public rad::RefCounted<CommandPool>
{
public:
    CommandPool(rad::Ref<Device> device, QueueFamily queueFamily);
    ~CommandPool();

    vk::CommandPool GetHandle() const { return static_cast<vk::CommandPool>(m_handle); }

    vk::raii::CommandBuffers Allocate(vk::CommandBufferLevel level, uint32_t count);

    rad::Ref<Device> m_device;
    QueueFamily m_queueFamily;
    vk::raii::CommandPool m_handle = { nullptr };

}; // class CommandPool

class CommandRecorder
{
public:
    vk::raii::CommandBuffer& m_cmdBuffer;

    CommandRecorder(vk::raii::CommandBuffer& cmdBuffer) :
        m_cmdBuffer(cmdBuffer)
    {
    }
    ~CommandRecorder()
    {
    }

    // https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples

    // Generally considered more efficient to do a global memory barrier than per-resource barriers,
    // per-resource barriers should usually be used for queue ownership transfers
    // and image layout transitions - otherwise use global barriers.

    void SetMemoryBarrier(
        vk::PipelineStageFlags2KHR srcStageMask, vk::AccessFlags2KHR srcAccessMask,
        vk::PipelineStageFlags2KHR dstStageMask, vk::AccessFlags2KHR dstAccessMask);
    void SetImageBarrier(
        vk::PipelineStageFlags2KHR srcStageMask, vk::AccessFlags2KHR srcAccessMask,
        vk::PipelineStageFlags2KHR dstStageMask, vk::AccessFlags2KHR dstAccessMask,
        vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
        vk::Image image, const vk::ImageSubresourceRange& range);

    // StorageBuffer/StorageImage Read-After-Write (RAW hazard)
    void SetMemoryBarrier_ComputeToComputeRAW();
    // StorageBuffer/StorageImage Write-After-Read (WAR hazard), requires execution dependency.
    void SetMemoryBarrier_ComputeToComputeWAR();

    // Dispatch writes into a storage buffer. Draw consumes that buffer as an index buffer.
    void SetMemoryBarrier_ComputeWriteToGraphicsIndexRead();
    // Dispatch writes into a storage buffer. Draw consumes that buffer as a draw indirect buffer.
    void SetMemoryBarrier_ComputeWriteToGraphicsIndirectCommandRead();
    // Dispatch writes into a storage image. Draw samples that image in a fragment shader.
    void SetImageBarrier_ComputeWriteToGraphicsSample(vk::Image image, const vk::ImageSubresourceRange& range);
    // Draw writes to a color attachment. Dispatch samples from that image.
    void SetImageBarrier_ColorAttachmentToComputeSample(vk::Image image, const vk::ImageSubresourceRange& range);
    // Draw writes to a depth attachment. Dispatch samples from that image.
    void SetImageBarrier_DepthStencilAttachmentToComputeSample(vk::Image image, const vk::ImageSubresourceRange& range);

    // First draw writes to a depth attachment. Second draw samples from that depth image in the fragment shader (e.g. shadow map rendering).
    void SetImageBarrier_DepthStencilAttachmentToFragmentSample(vk::Image image, const vk::ImageSubresourceRange& range);
    // First draw writes to a color attachment. Second draw samples from that color image in the fragment shader.
    void SetImageBarrier_ColorAttachmentToFragmentSample(vk::Image image, const vk::ImageSubresourceRange& range);
    // First draw samples a texture in the fragment shader. Second draw writes to that texture as a color attachment.
    // This is a WAR hazard, only requires layout transition.
    void SetImageBarrier_FragmentSampleToColorAttachment(vk::Image image, const vk::ImageSubresourceRange& range);

    // CPU read back of data written by a compute shader.
    void SetMemoryBarrier_ShaderWriteToHostRead(vk::PipelineStageFlagBits2 stage);

}; // class CommandRecorder

} // namespace vkpp
