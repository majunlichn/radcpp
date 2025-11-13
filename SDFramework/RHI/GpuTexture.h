#pragma once

#include <SDFramework/Core/Common.h>

namespace sdf
{

class GpuDevice;

class GpuTexture : public rad::RefCounted<GpuTexture>
{
public:
    static rad::Ref<GpuTexture> Create(rad::Ref<GpuDevice> device, const SDL_GPUTextureCreateInfo& createInfo);

    GpuTexture(rad::Ref<GpuDevice> device, SDL_GPUTexture* handle);
    ~GpuTexture();

    rad::Ref<GpuDevice> m_device;
    SDL_GPUTexture* m_handle;

}; // class GpuTexture

} // namespace sdf
