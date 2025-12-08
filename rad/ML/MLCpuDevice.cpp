#include <rad/ML/MLCpuDevice.h>
#include <thread>

namespace rad
{

MLCpuDevice::MLCpuDevice()
{
#if defined(CPU_FEATURES_ARCH_X86)
    m_name = StrTrim(g_X86Info.brand_string);
#endif
}

MLCpuDevice::~MLCpuDevice()
{
}

uint32_t MLCpuDevice::GetPhysicalCoreCount() const
{
    return static_cast<uint32_t>(GetNumberOfPhysicalCores());
}

uint32_t MLCpuDevice::GetLogicalCoreCount() const
{
    return static_cast<uint32_t>(std::thread::hardware_concurrency());
}

} // namespace rad
