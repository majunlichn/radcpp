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
class TensorStorage;

class Device : public rad::RefCounted<Device>
{
public:
    Device() = default;
    virtual ~Device() = default;

    DeviceType GetType() const { return m_type; }
    const std::string& GetName() const { return m_name; }

    virtual rad::Ref<Context> CreateContext() = 0;
    virtual rad::Ref<TensorStorage> CreateTensorStorage(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options = {}) = 0;

    virtual bool IsDataTypeSupported(DataType dataType) const = 0;
    virtual bool IsDataTypeComputable(DataType dataType) const = 0;

    DeviceType m_type = DeviceType::Unknown;
    std::string m_name;
    std::string m_driverVersion;

}; // class Device

} // namespace ML
