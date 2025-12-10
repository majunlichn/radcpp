#pragma once

#include <rad/ML/MLDataType.h>

#include <rad/Common/Algorithm.h>
#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>
#include <rad/Common/String.h>
#include <rad/Container/ArrayRef.h>
#include <rad/Container/SmallVector.h>
#include <rad/Container/Span.h>

namespace rad
{

enum class MLDeviceType
{
    Unknown,
    CPU,
    GPU,
    NPU,
};

class MLContext;
class MLTensor;

class MLDevice : public RefCounted<MLDevice>
{
public:
    MLDevice() = default;
    virtual ~MLDevice() = default;

    MLDeviceType GetType() const { return m_type; }

    virtual Ref<MLContext> CreateContext() = 0;
    virtual Ref<MLTensor> CreateTensor(MLDataType dataType, ArrayRef<size_t> sizes, ArrayRef<size_t> strides) = 0;

    MLDeviceType m_type = MLDeviceType::Unknown;
    std::string m_name;
    std::string m_driverVersion;

}; // class MLDevice

} // namespace rad
