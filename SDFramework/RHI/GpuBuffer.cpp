#include <SDFramework/RHI/GpuBuffer.h>
#include <SDFramework/RHI/GpuDevice.h>

namespace sdf
{

rad::Ref<GpuTransferBuffer> GpuTransferBuffer::Create(rad::Ref<GpuDevice> device, const SDL_GPUTransferBufferCreateInfo& createInfo)
{
    SDL_GPUTransferBuffer* bufferHandle = SDL_CreateGPUTransferBuffer(device->GetHandle(), &createInfo);
    if (bufferHandle)
    {
        return rad::Ref<GpuTransferBuffer>(RAD_NEW GpuTransferBuffer(device, bufferHandle));
    }
    else
    {
        SDF_LOG(err, "Failed to create a GpuTransferBuffer: {}", SDL_GetError());
        return nullptr;
    }
}

GpuTransferBuffer::GpuTransferBuffer(rad::Ref<GpuDevice> device, SDL_GPUTransferBuffer* handle) :
    m_device(std::move(device)),
    m_handle(handle)
{
}

GpuTransferBuffer::~GpuTransferBuffer()
{
    if (m_handle)
    {
        SDL_ReleaseGPUTransferBuffer(m_device->GetHandle(), m_handle);
        m_handle = nullptr;
    }
}

void* GpuTransferBuffer::Map(bool cycle)
{
    return SDL_MapGPUTransferBuffer(m_device->GetHandle(), m_handle, cycle);
}

void GpuTransferBuffer::Unmap()
{
    SDL_UnmapGPUTransferBuffer(m_device->GetHandle(), m_handle);
}

rad::Ref<GpuBuffer> GpuBuffer::Create(rad::Ref<GpuDevice> device, const SDL_GPUBufferCreateInfo& createInfo)
{
    SDL_GPUBuffer* bufferHandle = SDL_CreateGPUBuffer(device->GetHandle(), &createInfo);
    if (bufferHandle)
    {
        return rad::Ref<GpuBuffer>(RAD_NEW GpuBuffer(device, bufferHandle));
    }
    else
    {
        SDF_LOG(err, "Failed to create a GpuBuffer: {}", SDL_GetError());
        return nullptr;
    }
}

GpuBuffer::GpuBuffer(rad::Ref<GpuDevice> device, SDL_GPUBuffer* handle) :
    m_device(std::move(device)),
    m_handle(handle)
{
}

GpuBuffer::~GpuBuffer()
{
    if (m_handle)
    {
        SDL_ReleaseGPUBuffer(m_device->GetHandle(), m_handle);
        m_handle = nullptr;
    }
}

} // namespace sdf
