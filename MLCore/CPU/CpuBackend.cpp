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
    for (size_t i = 0; i < GetDeviceCount(); ++i)
    {
        Device* device = GetDevice(i);
        if (device != nullptr)
        {
            if (auto context = device->CreateContext())
            {
                SetDefaultContext(device, context);
            }
            else
            {
                ML_LOG(err, "Failed to create context for device#{}: {}", i, device->GetName());
                return false;
            }
        }
    }
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

rad::Ref<CpuBackend> CreateCpuBackend()
{
    rad::Ref<CpuBackend> backend = RAD_NEW CpuBackend();
    if (backend->Init())
    {
        ML_LOG(info, "CPU backend created.");
        for (size_t i = 0; i < backend->GetDeviceCount(); ++i)
        {
            Device* device = backend->GetDevice(i);
            ML_LOG(info, "Device#{}: {}", i, device->m_name, device->m_driverVersion);
        }
        return backend.get();
    }
    return nullptr;
}

} // namespace ML
