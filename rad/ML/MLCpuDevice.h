#pragma once

#include <rad/ML/MLDevice.h>
#include <rad/System/CpuInfo.h>

namespace rad
{

class MLContext;
class MLTensor;

// Backend: CPU
class MLCpuDevice : public MLDevice
{
public:
    MLCpuDevice();
    ~MLCpuDevice();

    uint32_t GetPhysicalCoreCount() const;
    uint32_t GetLogicalCoreCount() const;

    virtual Ref<MLContext> CreateContext() override;
    virtual Ref<MLTensor> CreateTensor(ArrayRef<size_t> sizes, MLDataType dataType, const MLTensorOptions& options = {}) override;

}; // class MLCpuDevice
 
} // namespace rad
