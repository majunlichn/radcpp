#pragma once

#include <SDFramework/Core/Common.h>

namespace sdf
{

class GpuDevice : public rad::RefCounted<GpuDevice>
{
public:
    // @param name: the preferred GPU driver, or nullptr to let SDL pick the optimal driver.
    static rad::Ref<GpuDevice> Create(SDL_GPUShaderFormat shaderFormat, bool enableDebugMode, const char* name);

    GpuDevice(SDL_GPUDevice* handle);
    ~GpuDevice();

    SDL_GPUDevice* GetHandle() const { return m_handle; }

    // Claims a window, creating a swapchain structure for it.
    bool ClaimWindow(SDL_Window* window);

    SDL_GPUDevice* m_handle;

}; // class GpuDevice

} // namespace sdf
