#include <SDFramework/RHI/GpuDevice.h>

namespace sdf
{

rad::Ref<GpuDevice> GpuDevice::Create(SDL_GPUShaderFormat shaderFormat, bool enableDebugMode, const char* name)
{
    SDL_GPUDevice* deviceHandle = SDL_CreateGPUDevice(shaderFormat, enableDebugMode, name);
    if (deviceHandle)
    {
        return rad::Ref<GpuDevice>(RAD_NEW GpuDevice(deviceHandle));
    }
    else
    {
        SDF_LOG(err, "Failed to create a GpuDevice: {}", SDL_GetError());
        return nullptr;
    }
}

GpuDevice::GpuDevice(SDL_GPUDevice* handle) :
    m_handle(handle)
{
}

GpuDevice::~GpuDevice()
{
    if (m_handle)
    {
        SDL_DestroyGPUDevice(m_handle);
        m_handle = nullptr;
    }
}

bool GpuDevice::ClaimWindow(SDL_Window* window)
{
    if (bool result = SDL_ClaimWindowForGPUDevice(m_handle, window))
    {
        return true;
    }
    else
    {
        SDF_LOG(err, "Failed to claim window for GpuDevice: {}", SDL_GetError());
        return false;
    }
}

} // namespace sdf
