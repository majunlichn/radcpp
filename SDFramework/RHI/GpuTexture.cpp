#include <SDFramework/RHI/GpuTexture.h>
#include <SDFramework/RHI/GpuDevice.h>

namespace sdf
{

rad::Ref<GpuTexture> GpuTexture::Create(rad::Ref<GpuDevice> device, const SDL_GPUTextureCreateInfo& createInfo)
{
    SDL_GPUTexture* textureHandle = SDL_CreateGPUTexture(device->GetHandle(), &createInfo);
    if (textureHandle)
    {
        return rad::Ref<GpuTexture>(RAD_NEW GpuTexture(device, textureHandle));
    }
    else
    {
        SDF_LOG(err, "Failed to create a GpuTexture: {}", SDL_GetError());
        return nullptr;
    }
}

GpuTexture::GpuTexture(rad::Ref<GpuDevice> device, SDL_GPUTexture* handle) :
    m_device(std::move(device)),
    m_handle(handle)
{
}

GpuTexture::~GpuTexture()
{
    if (m_handle)
    {
        SDL_ReleaseGPUTexture(m_device->GetHandle(), m_handle);
        m_handle = nullptr;
    }
}

} // namespace sdf
