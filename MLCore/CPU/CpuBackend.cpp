#include <MLCore/CPU/CpuBackend.h>
#include <MLCore/CPU/CpuDevice.h>
#include <MLCore/Global.h>
#include <MLCore/Logging.h>

namespace ML
{

CpuBackend::CpuBackend()
{
    m_device = RAD_NEW CpuDevice();
}

CpuBackend::~CpuBackend()
{
}

bool CpuBackend::Init()
{
    return true;
}

size_t CpuBackend::GetDeviceCount() const
{
    return 1;
}

Device* CpuBackend::GetDevice(size_t index)
{
    return m_device.get();
}

Backend* InitCpuBackend(std::string_view name)
{
    rad::Ref<CpuBackend> cpuBackend = RAD_NEW CpuBackend();
    if (cpuBackend->Init())
    {
        if (RegisterBackend("CPU", cpuBackend))
        {
            g_contextPool->CreateContextsForBackend(cpuBackend.get());
            ML_LOG(info, "CPU backend initialized.");
            return cpuBackend.get();
        }
    }
    return nullptr;
}

} // namespace ML
