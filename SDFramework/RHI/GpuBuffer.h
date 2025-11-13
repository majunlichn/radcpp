#pragma once

#include <SDFramework/Core/Common.h>

namespace sdf
{

class GpuDevice;

class GpuTransferBuffer : public rad::RefCounted<GpuTransferBuffer>
{
public:
    static rad::Ref<GpuTransferBuffer> Create(rad::Ref<GpuDevice> device, const SDL_GPUTransferBufferCreateInfo& createInfo);

    GpuTransferBuffer(rad::Ref<GpuDevice> device, SDL_GPUTransferBuffer* handle);
    ~GpuTransferBuffer();

    // @param cycle: if true, cycles the transfer buffer if it is already bound.
    void* Map(bool cycle);
    void Unmap();

    rad::Ref<GpuDevice> m_device;
    SDL_GPUTransferBuffer* m_handle;

}; // class GpuTransferBuffer

class GpuBuffer : public rad::RefCounted<GpuBuffer>
{
public:
    static rad::Ref<GpuBuffer> Create(rad::Ref<GpuDevice> device, const SDL_GPUBufferCreateInfo& createInfo);

    GpuBuffer(rad::Ref<GpuDevice> device, SDL_GPUBuffer* handle);
    ~GpuBuffer();

    rad::Ref<GpuDevice> m_device;
    SDL_GPUBuffer* m_handle;

}; // class GpuBuffer

} // namespace sdf
