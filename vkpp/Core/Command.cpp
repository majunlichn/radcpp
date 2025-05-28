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

std::vector<rad::Ref<CommandBuffer>> CommandPool::Allocate(vk::CommandBufferLevel level, uint32_t count)
{
    vk::CommandBufferAllocateInfo allocateInfo(m_wrapper, level, count);
    std::vector<vk::CommandBuffer> cmdBufferHandles(count);
    VK_CHECK(
        m_device->m_wrapper.getDispatcher()->vkAllocateCommandBuffers(
            m_device->GetHandle(),
            reinterpret_cast<const VkCommandBufferAllocateInfo*>(&allocateInfo),
            reinterpret_cast<VkCommandBuffer*>(cmdBufferHandles.data()))
    );
    std::vector<rad::Ref<CommandBuffer>> cmdBuffers(count);
    for (size_t i = 0; i < count; ++i)
    {
        cmdBuffers[i] = RAD_NEW CommandBuffer(this, cmdBufferHandles[i]);
    }
    return cmdBuffers;
}

CommandBuffer::CommandBuffer(
    rad::Ref<Device> device, vk::CommandPool cmdPoolHandle, vk::CommandBuffer cmdBufferHandle) :
    m_device(std::move(device))
{
    m_wrapper = vk::raii::CommandBuffer(m_device->m_wrapper, cmdBufferHandle, cmdPoolHandle);
}

CommandBuffer::CommandBuffer(
    rad::Ref<CommandPool> cmdPool, vk::CommandBuffer cmdBufferHandle) :
    m_device(cmdPool->m_device),
    m_cmdPool(std::move(cmdPool))
{
    m_wrapper = vk::raii::CommandBuffer(m_device->m_wrapper, cmdBufferHandle, m_cmdPool->GetHandle());
}

CommandBuffer::~CommandBuffer()
{
    m_wrapper.clear();
    m_cmdPool.reset();
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

void CommandBuffer::TransitLayout(
    Image* image,
    vk::PipelineStageFlags2     srcStageMask,
    vk::AccessFlags2            srcAccessMask,
    vk::PipelineStageFlags2     dstStageMask,
    vk::AccessFlags2            dstAccessMask,
    vk::ImageLayout             oldLayout,
    vk::ImageLayout             newLayout,
    const vk::ImageSubresourceRange* subresourceRange)
{
    vk::ImageMemoryBarrier2 imageBarrier = {};
    imageBarrier.srcStageMask = srcStageMask;
    imageBarrier.srcAccessMask = srcAccessMask;
    imageBarrier.dstStageMask = dstStageMask;
    imageBarrier.dstAccessMask = dstAccessMask;
    imageBarrier.oldLayout = oldLayout;
    imageBarrier.newLayout = newLayout;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = image->GetHandle();
    if (subresourceRange)
    {
        imageBarrier.subresourceRange = *subresourceRange;
    }
    else
    {
        imageBarrier.subresourceRange.aspectMask = GetImageAspectFromFormat(image->GetFormat());
        imageBarrier.subresourceRange.baseMipLevel = 0;
        imageBarrier.subresourceRange.levelCount = image->GetMipLevels();
        imageBarrier.subresourceRange.baseArrayLayer = 0;
        imageBarrier.subresourceRange.layerCount = image->GetArrayLayers();
    }

    vk::DependencyInfoKHR dependency;
    dependency.setImageMemoryBarriers(imageBarrier);
    this->SetPipelineBarrier2(dependency);

    image->SetCurrentPipelineStage(dstStageMask);
    image->SetCurrentAccessFlags(dstAccessMask);
    image->SetCurrentLayout(newLayout);
}

void CommandBuffer::TransitLayoutFromCurrent(
    Image* image,
    vk::PipelineStageFlags2     dstStageMask,
    vk::AccessFlags2            dstAccessMask,
    vk::ImageLayout             newLayout,
    const vk::ImageSubresourceRange* subresourceRange)
{
    TransitLayout(image,
        image->GetCurrentPipelineStage(), image->GetCurrentAccessMask(),
        dstStageMask, dstAccessMask,
        image->GetCurrentLayout(), newLayout,
        subresourceRange);
}

void CommandBuffer::SetImageBarrier_ColorAttachmentToComputeSample(
    Image* image, const vk::ImageSubresourceRange* range)
{
    TransitLayout(image,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderRead,
        vk::ImageLayout::eAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        range);
}

void CommandBuffer::SetImageBarrier_ColorAttachmentToFragmentSample(Image* image, const vk::ImageSubresourceRange* range)
{
    TransitLayout(image,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eShaderRead,
        vk::ImageLayout::eAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        range);
}

void CommandBuffer::SetImageBarrier_DepthStencilAttachmentToComputeSample(
    Image* image, const vk::ImageSubresourceRange* range)
{
    TransitLayout(image,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderRead,
        vk::ImageLayout::eAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        range);
}

void CommandBuffer::SetImageBarrier_DepthStencilAttachmentToFragmentSample(
    Image* image, const vk::ImageSubresourceRange* range)
{
    TransitLayout(image,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eShaderRead,
        vk::ImageLayout::eAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        range);
}

void CommandBuffer::SetImageBarrier_FragmentSampleToColorAttachment(
    Image* image, const vk::ImageSubresourceRange* range)
{
    TransitLayout(image,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2(),
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2(),
        vk::ImageLayout::eReadOnlyOptimal,
        vk::ImageLayout::eAttachmentOptimal,
        range);
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

void CommandBuffer::BeginRendering(const vk::RenderingInfo& renderingInfo)
{
    m_wrapper.beginRendering(renderingInfo);
}

void CommandBuffer::BeginRendering(
    const vk::Rect2D& renderArea,
    uint32_t layerCount,
    uint32_t viewMask,
    rad::ArrayRef<vk::RenderingAttachmentInfo> colorAttachments,
    const vk::RenderingAttachmentInfo* depthAttachment,
    const vk::RenderingAttachmentInfo* stencilAttachment)
{
    vk::RenderingInfoKHR renderingInfo = {};
    renderingInfo.renderArea = renderArea;
    renderingInfo.layerCount = layerCount;
    renderingInfo.viewMask = viewMask;
    renderingInfo.colorAttachmentCount = colorAttachments.size32();
    renderingInfo.pColorAttachments = colorAttachments.data();
    renderingInfo.pDepthAttachment = depthAttachment;
    renderingInfo.pStencilAttachment = stencilAttachment;
    BeginRendering(renderingInfo);
}

void CommandBuffer::EndRendering()
{
    m_wrapper.endRendering();
}

vk::RenderingAttachmentInfo MakeRenderingAttachmentInfo(
    ImageView* view, vk::AttachmentLoadOp loadOp, vk::AttachmentStoreOp storeOp, const vk::ClearValue& clearValue)
{
    vk::RenderingAttachmentInfo attachInfo = {};
    attachInfo.imageView = view->GetHandle();
    attachInfo.imageLayout = view->GetImage()->GetCurrentLayout();
    attachInfo.loadOp = loadOp;
    attachInfo.storeOp = storeOp;
    attachInfo.clearValue = clearValue;
    return attachInfo;
}

} // namespace vkpp
