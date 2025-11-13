#pragma once

#include <SDFramework/Core/Common.h>

namespace sdf
{

class GpuDevice;

class GpuCommandBuffer : public rad::RefCounted<GpuCommandBuffer>
{
public:
    static rad::Ref<GpuCommandBuffer> Create(rad::Ref<GpuDevice> device);

    GpuCommandBuffer(rad::Ref<GpuDevice> device, SDL_GPUCommandBuffer* handle);
    ~GpuCommandBuffer();

    SDL_GPURenderPass* BeginRenderPass(
        rad::ArrayRef<SDL_GPUColorTargetInfo> colorTargetInfos,
        const SDL_GPUDepthStencilTargetInfo* depthStencilTargetInfo = nullptr);
    void EndRenderPass(SDL_GPURenderPass* renderPass);

    SDL_GPUComputePass* BeginComputePass(
        rad::ArrayRef<SDL_GPUStorageTextureReadWriteBinding> storageTextureBindings,
        rad::ArrayRef<SDL_GPUStorageBufferReadWriteBinding> storageBufferBindings);
    void EndComputePass(SDL_GPUComputePass* computePass);

    void BindPipeline(SDL_GPURenderPass* renderPass, SDL_GPUGraphicsPipeline* pipeline);
    void BindPipeline(SDL_GPUComputePass* computePass, SDL_GPUComputePipeline* pipeline);
    void SetViewport(SDL_GPURenderPass* renderPass, const SDL_GPUViewport* viewport);
    void BindVertexBuffer(
        SDL_GPURenderPass* renderPass,
        Uint32 firstSlot,
        rad::ArrayRef<SDL_GPUBufferBinding> bindings);
    void BindVertexSamplers(
        SDL_GPURenderPass* renderPass,
        Uint32 firstSlot,
        rad::ArrayRef<SDL_GPUTextureSamplerBinding> bindings);

    void BindComputeStorageBuffers(
        SDL_GPUComputePass* computePass,
        Uint32 firstSlot,
        rad::ArrayRef<SDL_GPUBuffer*> storageBuffers);
    void BindComputeStorageTextures(
        SDL_GPUComputePass* computePass,
        Uint32 firstSlot,
        rad::ArrayRef<SDL_GPUTexture*> storageTextures);

    void DrawGPUPrimitives(
        SDL_GPURenderPass* renderPass,
        Uint32 vertexCount,
        Uint32 instanceCount,
        Uint32 firstVertex,
        Uint32 firstInstance);
    void DrawGPUPrimitivesIndirect(
        SDL_GPURenderPass* renderPass,
        SDL_GPUBuffer* buffer,
        Uint32 offset,
        Uint32 drawCount);
    void DrawGPUIndexedPrimitivesIndirect(
        SDL_GPURenderPass* renderPass,
        SDL_GPUBuffer* buffer,
        Uint32 offset,
        Uint32 drawCount);

    void DispatchCompute(
        SDL_GPUComputePass* computePass,
        Uint32 groupCountX,
        Uint32 groupCountY,
        Uint32 groupCountZ);

    bool Submit();

    rad::Ref<GpuDevice> m_device;
    SDL_GPUCommandBuffer* m_handle;

}; // class GpuCommandBuffer

} // namespace sdf
