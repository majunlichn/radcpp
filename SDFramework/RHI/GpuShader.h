#pragma once

#include <SDFramework/Core/Common.h>

namespace sdf
{

class GpuDevice;

class GpuShader : public rad::RefCounted<GpuShader>
{
public:
    static rad::Ref<GpuShader> Create(rad::Ref<GpuDevice> device, const SDL_GPUShaderCreateInfo& createInfo);

    GpuShader(rad::Ref<GpuDevice> device, SDL_GPUShader* handle);
    ~GpuShader();

    rad::Ref<GpuDevice> m_device;
    SDL_GPUShader* m_handle;

}; // class GpuShader

} // namespace sdf
