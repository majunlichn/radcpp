#pragma once

#include <MLCore/Device.h>

namespace ML
{

class Context;
class Tensor;

// Backend: CPU
class CpuDevice : public Device
{
public:
    CpuDevice();
    ~CpuDevice();

    uint32_t GetPhysicalCoreCount() const;
    uint32_t GetLogicalCoreCount() const;

    virtual rad::Ref<Context> CreateContext() override;
    virtual rad::Ref<TensorStorage> CreateTensorStorage(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options = {}) override;

    virtual bool IsDataTypeSupported(DataType dataType) const override;
    virtual bool IsDataTypeComputable(DataType dataType) const override;

}; // class CpuDevice

} // namespace ML
