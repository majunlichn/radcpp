#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Surface : public rad::RefCounted<Surface>
{
public:
    Surface(rad::Ref<Instance> instance, vk::SurfaceKHR surfaceHandle);
    ~Surface();

    vk::SurfaceKHR GetHandle() const { return m_wrapper; }

    rad::Ref<Instance> m_instance;
    vk::raii::SurfaceKHR m_wrapper = { nullptr };

}; // class Surface

} // namespace vkpp
