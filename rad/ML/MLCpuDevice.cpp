#include <rad/ML/MLCpuDevice.h>
#include <rad/ML/MLCpuContext.h>
#include <rad/ML/MLCpuTensor.h>
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

Ref<MLContext> MLCpuDevice::CreateContext()
{
    return RAD_NEW MLCpuContext(this);
}

Ref<MLTensor> MLCpuDevice::CreateTensor(MLDataType dataType, ArrayRef<size_t> sizes, ArrayRef<size_t> strides)
{
    Ref<MLCpuTensor> tensor = RAD_NEW MLCpuTensor(this);
    if (tensor->Init(dataType, sizes, strides))
    {
        return tensor;
    }
    else
    {
        return nullptr;
    }
}

} // namespace rad
