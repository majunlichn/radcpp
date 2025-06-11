#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class CommandPool : public rad::RefCounted<CommandPool>
{
public:
    CommandPool(rad::Ref<Device> device, QueueFamily queueFamily, vk::CommandPoolCreateFlags flags);
    ~CommandPool();

    vk::CommandPool GetHandle() const { return static_cast<vk::CommandPool>(m_wrapper); }
    const DeviceDispatcher* GetDispatcher() const;

    std::vector<rad::Ref<CommandBuffer>> Allocate(vk::CommandBufferLevel level, uint32_t count);
    std::vector<rad::Ref<CommandBuffer>> AllocatePrimary(uint32_t count)
    {
        return Allocate(vk::CommandBufferLevel::ePrimary, count);
    }
    std::vector<rad::Ref<CommandBuffer>> AllocateSecondary(uint32_t count)
    {
        return Allocate(vk::CommandBufferLevel::eSecondary, count);
    }

    rad::Ref<Device> m_device;
    QueueFamily m_queueFamily;
    vk::raii::CommandPool m_wrapper = { nullptr };

}; // class CommandPool

class CommandBuffer : public rad::RefCounted<CommandBuffer>
{
public:
    rad::Ref<Device> m_device;
    rad::Ref<CommandPool> m_cmdPool;
    vk::raii::CommandBuffer m_wrapper = { nullptr };

    CommandBuffer(rad::Ref<Device> device, vk::CommandPool cmdPoolHandle, vk::CommandBuffer cmdBufferHandle);
    CommandBuffer(rad::Ref<CommandPool> cmdPool, vk::CommandBuffer cmdBufferHandle);
    ~CommandBuffer();

    vk::CommandBuffer GetHandle() const { return m_wrapper; }
    const DeviceDispatcher* GetDispatcher() const;

    void Begin(
        vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr);
    void End();
    void Reset(vk::CommandBufferResetFlags flags) { m_wrapper.reset(flags); }

    void BindPipeline(vk::PipelineBindPoint bindPoint, vk::Pipeline pipeline)
    {
        m_wrapper.bindPipeline(bindPoint, pipeline);
    }
    void BindPipeine(Pipeline* pipeline);

    void SetViewport(uint32_t firstViewport, rad::ArrayRef<vk::Viewport> viewports)
    {
        m_wrapper.setViewport(firstViewport, viewports);
    }

    void SetScissor(uint32_t firstScissor, rad::ArrayRef<vk::Rect2D> scissors)
    {
        m_wrapper.setScissor(firstScissor, scissors);
    }

    void SetLineWidth(float lineWidth)
    {
        m_wrapper.setLineWidth(lineWidth);
    }

    void SetDepthBias(float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor)
    {
        m_wrapper.setDepthBias(depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor);
    }

    void SetBlendConstants(rad::ArrayRef<float> blendConstants)
    {
        assert(blendConstants.size() == 4);
        m_wrapper.setBlendConstants(blendConstants.data());
    }

    void SetDepthBounds(float minDepthBounds, float maxDepthBounds)
    {
        m_wrapper.setDepthBounds(minDepthBounds, maxDepthBounds);
    }

    void SetStencilCompareMask(vk::StencilFaceFlags faceMask, uint32_t compareMask)
    {
        m_wrapper.setStencilCompareMask(faceMask, compareMask);
    }

    void SetStencilWriteMask(vk::StencilFaceFlags faceMask, uint32_t writeMask)
    {
        m_wrapper.setStencilWriteMask(faceMask, writeMask);
    }

    void SetStencilReference(vk::StencilFaceFlags faceMask, uint32_t reference)
    {
        m_wrapper.setStencilReference(faceMask, reference);
    }

    void BindDescriptorSets(
        vk::PipelineBindPoint bindPoint,
        vk::PipelineLayout layout,
        uint32_t firstSet, rad::ArrayRef<vk::DescriptorSet> descriptorSets,
        rad::ArrayRef<uint32_t> dynamicOffsets = {})
    {
        m_wrapper.bindDescriptorSets(bindPoint, layout, firstSet, descriptorSets, dynamicOffsets);
    }

    void BindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType indexType)
    {
        m_wrapper.bindIndexBuffer(buffer, offset, indexType);
    }

    void BindVertexBuffer(uint32_t firstBinding, rad::ArrayRef<vk::Buffer> buffers, rad::ArrayRef<vk::DeviceSize> offsets)
    {
        m_wrapper.bindVertexBuffers(firstBinding, buffers, offsets);
    }

    void Draw(
        uint32_t vertexCount, uint32_t instanceCount,
        uint32_t firstVertex, uint32_t firstInstance)
    {
        m_wrapper.draw(vertexCount, instanceCount, firstVertex, firstInstance);
    }

    void DrawIndexed(
        uint32_t indexCount, uint32_t instanceCount,
        uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
    {
        m_wrapper.drawIndexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    }

    void DrawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t drawCount, uint32_t stride)
    {
        m_wrapper.drawIndirect(buffer, offset, drawCount, stride);
    }

    void DrawIndexedIndirect(
        vk::Buffer buffer, vk::DeviceSize offset, uint32_t drawCount, uint32_t stride)
    {
        m_wrapper.drawIndexedIndirect(buffer, offset, drawCount, stride);
    }

    void Dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
    {
        m_wrapper.dispatch(groupCountX, groupCountY, groupCountZ);
    }

    void DispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset)
    {
        m_wrapper.dispatchIndirect(buffer, offset);
    }

    void CopyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
        rad::ArrayRef<vk::BufferCopy> regions)
    {
        m_wrapper.copyBuffer(srcBuffer, dstBuffer, regions);
    }

    void CopyImage(
        vk::Image srcImage, vk::ImageLayout srcImageLayout,
        vk::Image dstImage, vk::ImageLayout dstImageLayout,
        rad::ArrayRef<vk::ImageCopy> regions)
    {
        m_wrapper.copyImage(srcImage, srcImageLayout, dstImage, dstImageLayout, regions);
    }

    void BlitImage(
        vk::Image srcImage, vk::ImageLayout srcImageLayout,
        vk::Image dstImage, vk::ImageLayout dstImageLayout,
        rad::ArrayRef<vk::ImageBlit> regions, vk::Filter filter)
    {
        m_wrapper.blitImage(srcImage, srcImageLayout, dstImage, dstImageLayout, regions, filter);
    }

    void CopyBufferToImage(
        vk::Buffer srcBuffer, vk::Image dstImage, vk::ImageLayout dstImageLayout,
        rad::ArrayRef<vk::BufferImageCopy> regions)
    {
        m_wrapper.copyBufferToImage(srcBuffer, dstImage, dstImageLayout, regions);
    }

    void CopyImageToBuffer(
        vk::Image srcImage, vk::ImageLayout srcImageLayout,
        vk::Buffer dstBuffer, rad::ArrayRef<vk::BufferImageCopy> regions)
    {
        m_wrapper.copyImageToBuffer(srcImage, srcImageLayout, dstBuffer, regions);
    }

    void UpdateBuffer(
        vk::Buffer dstBuffer, vk::DeviceSize dstOffset,
        const void* data, vk::DeviceSize size)
    {
        assert(dstOffset % 4 == 0);
        assert(size % 4 == 0);
        GetDispatcher()->vkCmdUpdateBuffer(
            static_cast<VkCommandBuffer>(GetHandle()),
            static_cast<VkBuffer>(dstBuffer),
            static_cast<VkDeviceSize>(dstOffset),
            size, data);
    }

    template <rad::TriviallyCopyable T>
    void UpdateBuffer(
        vk::Buffer dstBuffer, vk::DeviceSize dstOffset,
        rad::ArrayRef<T> elements)
    {
        assert(dstOffset % 4 == 0);
        assert((sizeof(T) * elements.size()) % 4 == 0);
        m_wrapper.updateBuffer(dstBuffer, dstOffset, elements);
    }

    void FillBuffer(vk::Buffer dstBuffer, vk::DeviceSize dstOffset, vk::DeviceSize size, uint32_t data)
    {
        m_wrapper.fillBuffer(dstBuffer, dstOffset, size, data);
    }

    void ClearColorImage(vk::Image image, vk::ImageLayout imageLayout,
        const vk::ClearColorValue& color, rad::ArrayRef<vk::ImageSubresourceRange> ranges)
    {
        m_wrapper.clearColorImage(image, imageLayout, color, ranges);
    }

    void ClearDepthStencilImage(vk::Image image, vk::ImageLayout imageLayout,
        const vk::ClearDepthStencilValue& depthStencil, rad::ArrayRef<vk::ImageSubresourceRange> ranges)
    {
        m_wrapper.clearDepthStencilImage(image, imageLayout, depthStencil, ranges);
    }

    void ClearAttachments(
        rad::ArrayRef<vk::ClearAttachment> attachments,
        rad::ArrayRef<vk::ClearRect> rects)
    {
        m_wrapper.clearAttachments(attachments, rects);
    }

    void ResolveImage(
        vk::Image srcImage, vk::ImageLayout srcImageLayout,
        vk::Image dstImage, vk::ImageLayout dstImageLayout,
        rad::ArrayRef<vk::ImageResolve> regions)
    {
        m_wrapper.resolveImage(srcImage, srcImageLayout, dstImage, dstImageLayout, regions);
    }

    void SetEvent(vk::Event event, vk::PipelineStageFlags stageMask)
    {
        m_wrapper.setEvent(event, stageMask);
    }

    void ResetEvent(vk::Event event, vk::PipelineStageFlags stageMask)
    {
        m_wrapper.resetEvent(event, stageMask);
    }

    void WaitEvents(
        rad::ArrayRef<vk::Event> events,
        vk::PipelineStageFlags srcStageMask,
        vk::PipelineStageFlags dstStageMask,
        rad::ArrayRef<vk::MemoryBarrier> memoryBarriers,
        rad::ArrayRef<vk::BufferMemoryBarrier> bufferMemoryBarriers,
        rad::ArrayRef<vk::ImageMemoryBarrier> imageMemoryBarriers)
    {
        m_wrapper.waitEvents(events, srcStageMask, dstStageMask, memoryBarriers, bufferMemoryBarriers, imageMemoryBarriers);
    }

    void SetPipelineBarrier(
        vk::PipelineStageFlags srcStageMask, vk::PipelineStageFlags dstStageMask,
        vk::DependencyFlags dependencyFlags,
        rad::ArrayRef<vk::MemoryBarrier> memoryBarriers,
        rad::ArrayRef<vk::BufferMemoryBarrier> bufferMemoryBarriers,
        rad::ArrayRef<vk::ImageMemoryBarrier> imageMemoryBarriers)
    {
        m_wrapper.pipelineBarrier(srcStageMask, dstStageMask, dependencyFlags, memoryBarriers, bufferMemoryBarriers, imageMemoryBarriers);
    }

    void SetPipelineBarrier2(vk::DependencyInfoKHR dependencyInfo)
    {
        m_wrapper.pipelineBarrier2(dependencyInfo);
    }

    void SetPipelineBarrier2(
        vk::DependencyFlags flags,
        rad::ArrayRef<vk::MemoryBarrier2> memoryBarriers,
        rad::ArrayRef<vk::BufferMemoryBarrier2> bufferMemoryBarriers,
        rad::ArrayRef<vk::ImageMemoryBarrier2> imageMemoryBarriers)
    {
        vk::DependencyInfoKHR dependencyInfo;
        dependencyInfo.setDependencyFlags(flags);
        dependencyInfo.setMemoryBarriers(memoryBarriers);
        dependencyInfo.setBufferMemoryBarriers(bufferMemoryBarriers);
        dependencyInfo.setImageMemoryBarriers(imageMemoryBarriers);
        m_wrapper.pipelineBarrier2(dependencyInfo);
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

    void TransitLayout(
        Image* image,
        vk::PipelineStageFlags2     srcStageMask,
        vk::AccessFlags2            srcAccessMask,
        vk::PipelineStageFlags2     dstStageMask,
        vk::AccessFlags2            dstAccessMask,
        vk::ImageLayout             oldLayout,
        vk::ImageLayout             newLayout,
        const vk::ImageSubresourceRange* subresourceRange = nullptr);
    void TransitLayoutFromCurrent(
        Image* image,
        vk::PipelineStageFlags2     dstStageMask,
        vk::AccessFlags2            dstAccessMask,
        vk::ImageLayout             newLayout,
        const vk::ImageSubresourceRange* subresourceRange = nullptr);

    // Draw writes to a color attachment. Dispatch samples from that image.
    void SetImageBarrier_ColorAttachmentToComputeSample(Image* image, const vk::ImageSubresourceRange* range = nullptr);
    // First draw writes to a color attachment. Second draw samples from that color image in the fragment shader.
    void SetImageBarrier_ColorAttachmentToFragmentSample(Image* image, const vk::ImageSubresourceRange* range = nullptr);
    // Draw writes to a depth attachment. Dispatch samples from that image.
    void SetImageBarrier_DepthStencilAttachmentToComputeSample(Image* image, const vk::ImageSubresourceRange* range = nullptr);
    // First draw writes to a depth attachment. Second draw samples from that depth image in the fragment shader (e.g. shadow map rendering).
    void SetImageBarrier_DepthStencilAttachmentToFragmentSample(Image* image, const vk::ImageSubresourceRange* range = nullptr);

    // First draw samples a texture in the fragment shader. Second draw writes to that texture as a color attachment.
    // This is a WAR hazard, only requires layout transition.
    void SetImageBarrier_FragmentSampleToColorAttachment(Image* image, const vk::ImageSubresourceRange* range = nullptr);

    // CPU read back of data written by shaders.
    void SetMemoryBarrier_ShaderWriteToHostRead(vk::PipelineStageFlagBits2 stage);

    // Query
    void BeginQuery(vk::QueryPool queryPool, uint32_t firstQuery, vk::QueryControlFlags flags = {})
    {
        m_wrapper.beginQuery(queryPool, firstQuery, flags);
    }

    void EndQuery(vk::QueryPool queryPool, uint32_t firstQuery)
    {
        m_wrapper.endQuery(queryPool, firstQuery);
    }

    void ResetQueryPool(vk::QueryPool queryPool, uint32_t firstQuery, uint32_t queryCount)
    {
        m_wrapper.resetQueryPool(queryPool, firstQuery, queryCount);
    }

    void WriteTimestamp(
        vk::PipelineStageFlagBits stage, vk::QueryPool queryPool, uint32_t query)
    {
        m_wrapper.writeTimestamp(stage, queryPool, query);
    }

    void CopyQueryPoolResults(
        vk::QueryPool queryPool, uint32_t firstQuery, uint32_t queryCount,
        vk::Buffer dstBuffer, vk::DeviceSize dstOffset, vk::DeviceSize stride,
        vk::QueryResultFlags flags = {})
    {
        m_wrapper.copyQueryPoolResults(queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags);
    }

    void SetPushConstants(
        vk::PipelineLayout layout, vk::ShaderStageFlags stageFlags,
        uint32_t offset, uint32_t size, const void* pValues)
    {
        GetDispatcher()->vkCmdPushConstants(
            static_cast<VkCommandBuffer>(GetHandle()),
            static_cast<VkPipelineLayout>(layout),
            static_cast<VkShaderStageFlags>(stageFlags),
            offset, size, pValues);
    }

    template <typename T>
    void SetPushConstants(
        vk::PipelineLayout layout, vk::ShaderStageFlags stageFlags,
        uint32_t offset, rad::ArrayRef<T> values)
    {
        assert(offset % 4 == 0);
        assert((sizeof(T) * values.size()) % 4 == 0);
        m_wrapper.pushConstants(layout, stageFlags, offset, values);
    }

    void BeginRenderPass(
        vk::RenderPass renderPass, vk::Framebuffer framebuffer,
        const vk::Rect2D& renderArea,
        rad::ArrayRef<vk::ClearValue> clearValues,
        vk::SubpassContents contents = vk::SubpassContents::eInline)
    {
        vk::RenderPassBeginInfo beginInfo;
        beginInfo.renderPass = renderPass;
        beginInfo.framebuffer = framebuffer;
        beginInfo.renderArea = renderArea;
        beginInfo.setClearValues(clearValues);
        m_wrapper.beginRenderPass(beginInfo, contents);
    }

    void NextSubpass(vk::SubpassContents contents = vk::SubpassContents::eInline)
    {
        m_wrapper.nextSubpass(contents);
    }

    void EndRenderPass()
    {
        m_wrapper.endRenderPass();
    }

    void BeginRendering(const vk::RenderingInfo& renderingInfo);
    void BeginRendering(
        const vk::Rect2D& renderArea,
        uint32_t layerCount,
        uint32_t viewMask,
        rad::ArrayRef<vk::RenderingAttachmentInfo> colorAttachments,
        const vk::RenderingAttachmentInfo* depthAttachment = nullptr,
        const vk::RenderingAttachmentInfo* stencilAttachment = nullptr);
    void EndRendering();

    void ExecuteCommands(rad::ArrayRef<vk::CommandBuffer> cmdBuffers)
    {
        m_wrapper.executeCommands(cmdBuffers);
    }

}; // class CommandBuffer

vk::RenderingAttachmentInfo MakeRenderingAttachmentInfo(
    ImageView* view, vk::AttachmentLoadOp loadOp, vk::AttachmentStoreOp storeOp, const vk::ClearValue& clearValue);

} // namespace vkpp
