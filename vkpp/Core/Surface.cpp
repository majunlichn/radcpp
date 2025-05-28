#include <vkpp/Core/Surface.h>
#include <vkpp/Core/Instance.h>

namespace vkpp
{

Surface::Surface(rad::Ref<Instance> instance, vk::SurfaceKHR surfaceHandle) :
    m_instance(std::move(instance))
{
    m_wrapper = vk::raii::SurfaceKHR(m_instance->m_wrapper, surfaceHandle);
}

Surface::~Surface()
{
}

} // namespace vkpp
