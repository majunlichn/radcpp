#include <SDFramework/RHI/GpuShader.h>
#include <SDFramework/RHI/GpuDevice.h>

namespace sdf
{

rad::Ref<GpuShader> GpuShader::Create(rad::Ref<GpuDevice> device, const SDL_GPUShaderCreateInfo& createInfo)
{
    SDL_GPUShader* shaderHandle = SDL_CreateGPUShader(device->GetHandle(), &createInfo);
    if (shaderHandle)
    {
        return rad::Ref<GpuShader>(RAD_NEW GpuShader(device, shaderHandle));
    }
    else
    {
        SDF_LOG(err, "Failed to create a GpuShader: {}", SDL_GetError());
        return nullptr;
    }
}

GpuShader::GpuShader(rad::Ref<GpuDevice> device, SDL_GPUShader* handle) :
    m_device(std::move(device)),
    m_handle(handle)
{
}

GpuShader::~GpuShader()
{
    if (m_handle)
    {
        SDL_ReleaseGPUShader(m_device->GetHandle(), m_handle);
        m_handle = nullptr;
    }
}

} // namespace sdf
