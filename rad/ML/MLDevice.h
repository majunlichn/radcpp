#pragma once

#include <rad/ML/MLDataType.h>
#include <rad/ML/MLTensorOptions.h>

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
    MLDevice(std::string_view backend) : m_backend(backend) {}
    virtual ~MLDevice() = default;

    MLDeviceType GetType() const { return m_type; }

    virtual Ref<MLContext> CreateContext() = 0;
    virtual Ref<MLTensor> CreateTensor(ArrayRef<size_t> sizes, MLDataType dataType, const MLTensorOptions& options = {}) = 0;
    // Create a tensor that has the same data type, sizes and strides.
    Ref<MLTensor> CreateTensorLike(MLTensor* input);

    MLDeviceType m_type = MLDeviceType::Unknown;
    std::string m_backend;
    std::string m_name;
    std::string m_driverVersion;

}; // class MLDevice

MLDevice* MLRegisterGlobalDevice(std::string_view backend, Ref<MLDevice> device);
MLDevice* MLGetGlobalDevice(std::string_view backend);
MLContext* MLRegisterPerThreadContext(std::string_view backend, Ref<MLContext> context);
MLContext* MLGetPerThreadContext(std::string_view backend);

Ref<MLTensor> MLCreateTensor(ArrayRef<size_t> sizes, MLDataType dataType, std::string_view backend = "CPU", const MLTensorOptions& options = {});

} // namespace rad
