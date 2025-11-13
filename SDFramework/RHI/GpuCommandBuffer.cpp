#include <SDFramework/RHI/GpuCommandBuffer.h>
#include <SDFramework/RHI/GpuDevice.h>

namespace sdf
{

rad::Ref<GpuCommandBuffer> GpuCommandBuffer::Create(rad::Ref<GpuDevice> device)
{
    SDL_GPUCommandBuffer* cmdBufferHandle = SDL_AcquireGPUCommandBuffer(device->GetHandle());
    if (cmdBufferHandle)
    {
        return rad::Ref<GpuCommandBuffer>(RAD_NEW GpuCommandBuffer(device, cmdBufferHandle));
    }
    else
    {
        SDF_LOG(err, "Failed to acquire a GpuCommandBuffer: {}", SDL_GetError());
        return nullptr;
    }
}

GpuCommandBuffer::GpuCommandBuffer(rad::Ref<GpuDevice> device, SDL_GPUCommandBuffer* handle) :
    m_device(std::move(device)),
    m_handle(handle)
{
}

GpuCommandBuffer::~GpuCommandBuffer()
{
    if (m_handle)
    {
        m_handle = nullptr;
    }
}

SDL_GPURenderPass* GpuCommandBuffer::BeginRenderPass(
    rad::ArrayRef<SDL_GPUColorTargetInfo> colorTargetInfos,
    const SDL_GPUDepthStencilTargetInfo* depthStencilTargetInfo)
{
    return SDL_BeginGPURenderPass(
        m_handle, colorTargetInfos.data(), static_cast<Uint32>(colorTargetInfos.size32()), depthStencilTargetInfo);
}

void GpuCommandBuffer::EndRenderPass(SDL_GPURenderPass* renderPass)
{
    SDL_EndGPURenderPass(renderPass);
}

SDL_GPUComputePass* GpuCommandBuffer::BeginComputePass(
    rad::ArrayRef<SDL_GPUStorageTextureReadWriteBinding> storageTextureBindings,
    rad::ArrayRef<SDL_GPUStorageBufferReadWriteBinding> storageBufferBindings)
{
    return SDL_BeginGPUComputePass(m_handle,
        storageTextureBindings.data(), static_cast<Uint32>(storageTextureBindings.size32()),
        storageBufferBindings.data(), static_cast<Uint32>(storageBufferBindings.size32()));
}

void GpuCommandBuffer::EndComputePass(SDL_GPUComputePass* computePass)
{
    SDL_EndGPUComputePass(computePass);
}

void GpuCommandBuffer::BindPipeline(SDL_GPURenderPass* renderPass, SDL_GPUGraphicsPipeline* pipeline)
{
    SDL_BindGPUGraphicsPipeline(renderPass, pipeline);
}

void GpuCommandBuffer::BindPipeline(SDL_GPUComputePass* computePass, SDL_GPUComputePipeline* pipeline)
{
    SDL_BindGPUComputePipeline(computePass, pipeline);
}

void GpuCommandBuffer::SetViewport(SDL_GPURenderPass* renderPass, const SDL_GPUViewport* viewport)
{
    SDL_SetGPUViewport(renderPass, viewport);
}

void GpuCommandBuffer::BindVertexBuffer(SDL_GPURenderPass* renderPass, Uint32 firstSlot, rad::ArrayRef<SDL_GPUBufferBinding> bindings)
{
    SDL_BindGPUVertexBuffers(renderPass, firstSlot, bindings.data(), static_cast<Uint32>(bindings.size32()));
}

void GpuCommandBuffer::BindVertexSamplers(SDL_GPURenderPass* renderPass, Uint32 firstSlot, rad::ArrayRef<SDL_GPUTextureSamplerBinding> bindings)
{
    SDL_BindGPUVertexSamplers(renderPass, firstSlot, bindings.data(), static_cast<Uint32>(bindings.size32()));
}

void GpuCommandBuffer::BindComputeStorageBuffers(SDL_GPUComputePass* computePass, Uint32 firstSlot, rad::ArrayRef<SDL_GPUBuffer*> storageBuffers)
{
    SDL_BindGPUComputeStorageBuffers(computePass, firstSlot, storageBuffers.data(), static_cast<Uint32>(storageBuffers.size32()));
}

void GpuCommandBuffer::BindComputeStorageTextures(SDL_GPUComputePass* computePass, Uint32 firstSlot, rad::ArrayRef<SDL_GPUTexture*> storageTextures)
{
    SDL_BindGPUComputeStorageTextures(computePass, firstSlot, storageTextures.data(), static_cast<Uint32>(storageTextures.size32()));
}

void GpuCommandBuffer::DrawGPUPrimitives(SDL_GPURenderPass* renderPass, Uint32 vertexCount, Uint32 instanceCount, Uint32 firstVertex, Uint32 firstInstance)
{
    SDL_DrawGPUPrimitives(renderPass, vertexCount, instanceCount, firstVertex, firstInstance);
}

void GpuCommandBuffer::DrawGPUPrimitivesIndirect(SDL_GPURenderPass* renderPass, SDL_GPUBuffer* buffer, Uint32 offset, Uint32 drawCount)
{
    SDL_DrawGPUPrimitivesIndirect(renderPass, buffer, offset, drawCount);
}

void GpuCommandBuffer::DrawGPUIndexedPrimitivesIndirect(SDL_GPURenderPass* renderPass, SDL_GPUBuffer* buffer, Uint32 offset, Uint32 drawCount)
{
    SDL_DrawGPUIndexedPrimitivesIndirect(renderPass, buffer, offset, drawCount);
}

void GpuCommandBuffer::DispatchCompute(
    SDL_GPUComputePass* computePass,
    Uint32 groupCountX,
    Uint32 groupCountY,
    Uint32 groupCountZ)
{
    SDL_DispatchGPUCompute(computePass, groupCountX, groupCountY, groupCountZ);
}

bool GpuCommandBuffer::Submit()
{
    return SDL_SubmitGPUCommandBuffer(m_handle);
}

} // namespace sdf
