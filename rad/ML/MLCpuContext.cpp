#include <rad/ML/MLCpuContext.h>
#include <rad/ML/MLCpuDevice.h>

namespace rad
{

MLCpuContext::MLCpuContext(Ref<MLCpuDevice> device) :
    m_device(std::move(device))
{
}

MLCpuContext::~MLCpuContext()
{
}

} // namespace rad
