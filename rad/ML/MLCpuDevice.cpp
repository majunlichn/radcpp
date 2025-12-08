#include <rad/ML/MLCpuDevice.h>
#include <thread>

namespace rad
{

MLCpuDevice::MLCpuDevice()
{
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
