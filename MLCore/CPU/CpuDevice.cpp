#include <MLCore/CPU/CpuDevice.h>
#include <MLCore/CPU/CpuContext.h>
#include <MLCore/CPU/CpuTensor.h>

#include <rad/System/CpuInfo.h>
#include <thread>

namespace ML
{

CpuDevice::CpuDevice()
{
#if defined(CPU_FEATURES_ARCH_X86)
    m_name = rad::StrTrim(rad::g_X86Info.brand_string);
#endif
}

CpuDevice::~CpuDevice()
{
}

uint32_t CpuDevice::GetPhysicalCoreCount() const
{
    return static_cast<uint32_t>(rad::GetNumberOfPhysicalCores());
}

uint32_t CpuDevice::GetLogicalCoreCount() const
{
    return static_cast<uint32_t>(std::thread::hardware_concurrency());
}

rad::Ref<Context> CpuDevice::CreateContext()
{
    return RAD_NEW CpuContext(this);
}

rad::Ref<TensorStorage> CpuDevice::CreateTensorStorage(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options)
{
    if (dataType == DataType::Unknown)
    {
        dataType = DataType::Float32;
    }
    rad::Ref<CpuTensorStorage> storage = RAD_NEW CpuTensorStorage(this);
    if (storage->Init(sizes, dataType, options))
    {
        return storage;
    }
    else
    {
        return nullptr;
    }
}

bool CpuDevice::IsDataTypeSupported(DataType dataType) const
{
    return true;    // CPU supports all data types, even non-native types can be emulated.
}

bool CpuDevice::IsDataTypeComputable(DataType dataType) const
{
    if ((dataType == DataType::Float32) ||
        (dataType == DataType::Float64) ||
        (dataType == DataType::Sint8) ||
        (dataType == DataType::Sint16) ||
        (dataType == DataType::Sint32) ||
        (dataType == DataType::Sint64) ||
        (dataType == DataType::Uint8) ||
        (dataType == DataType::Uint16) ||
        (dataType == DataType::Uint32) ||
        (dataType == DataType::Uint64))
    {
        return true;
    }
    else
    {
        return false;
    }
}

} // namespace ML
