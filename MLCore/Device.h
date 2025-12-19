#pragma once

#include <MLCore/Common.h>
#include <MLCore/Global.h>

namespace ML
{

enum class DeviceType
{
    Unknown,
    CPU,
    GPU,
    NPU,
};

class Context;
class Tensor;

class Device : public rad::RefCounted<Device>
{
public:
    Device() = default;
    virtual ~Device() = default;

    DeviceType GetType() const { return m_type; }

    virtual rad::Ref<Context> CreateContext() = 0;
    virtual rad::Ref<Tensor> CreateTensor(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options = {}) = 0;
    // Create a tensor that has the same data type, sizes and strides.
    rad::Ref<Tensor> CreateTensorLike(Tensor* input);

    virtual bool IsDataTypeSupported(DataType dataType) const = 0;
    virtual bool IsDataTypeComputable(DataType dataType) const = 0;

    DeviceType m_type = DeviceType::Unknown;
    std::string m_name;
    std::string m_driverVersion;

}; // class Device

rad::Ref<Tensor> CreateTensor(rad::ArrayRef<size_t> sizes, DataType dataType, Device* device = GetCurrentDevice(), const TensorOptions& options = {});

} // namespace ML
