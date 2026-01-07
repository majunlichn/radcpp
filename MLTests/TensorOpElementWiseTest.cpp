#include <MLCore/Backend.h>
#include <MLCore/Device.h>
#include <MLCore/Context.h>
#include <MLCore/Tensor.h>
#include <MLCore/Logging.h>
#include <MLCore/Vulkan/VulkanBackend.h>

#include <gtest/gtest.h>

extern std::vector<ML::Backend*> g_backends;

template <typename T>
void TestTensorOpAdd(ML::DataType dataType, ML::Backend* backend)
{
    ML::Device* device = backend->GetDevice(0);
    ML::SetCurrentDevice(device);
    if (!device || !device->IsDataTypeSupported(dataType))
    {
        ML_LOG(info, "{}.Add({}): not supported!", backend->m_name, ML::GetDataTypeName(dataType));
        return;
    }

    ML_LOG(info, "{}.Add({}): start", backend->m_name, ML::GetDataTypeName(dataType));

    static_assert(rad::is_floating_point_v<T> || std::is_integral_v<T>);
    using ComputeType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    a.FillConstant(ComputeType(1));
    b.FillConstant(ComputeType(1));
    ML::Tensor c = a.Add(b, ComputeType(2));

    c.AddScalarInPlace(ComputeType(1));

    std::vector<uint8_t> dataBuffer;
    dataBuffer.resize(c.GetDataSize());
    c.Read(dataBuffer.data(), 0, dataBuffer.size());
    const T* results = reinterpret_cast<const T*>(dataBuffer.data());
    for (size_t i = 0; i < c.GetElementCount(); ++i)
    {
        ASSERT_EQ(results[i], 4);
    }
}

template <typename T>
void TestTensorOpSubtract(ML::DataType dataType, ML::Backend* backend)
{
    ML::Device* device = backend->GetDevice(0);
    ML::SetCurrentDevice(device);
    if (!device || !device->IsDataTypeSupported(dataType))
    {
        ML_LOG(info, "{}.Subtract({}): not supported!", backend->m_name, ML::GetDataTypeName(dataType));
        return;
    }

    ML_LOG(info, "{}.Subtract({}): start", backend->m_name, ML::GetDataTypeName(dataType));

    static_assert(rad::is_floating_point_v<T> || std::is_integral_v<T>);
    using ComputeType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    a.FillConstant(ComputeType(4));
    b.FillConstant(ComputeType(1));
    ML::Tensor c = a.Subtract(b, ComputeType(2));

    c.SubtractScalarInPlace(ComputeType(1));

    std::vector<uint8_t> dataBuffer;
    dataBuffer.resize(c.GetDataSize());
    c.Read(dataBuffer.data(), 0, dataBuffer.size());
    const T* results = reinterpret_cast<const T*>(dataBuffer.data());
    for (size_t i = 0; i < c.GetElementCount(); ++i)
    {
        ASSERT_EQ(results[i], 1);
    }
}

template <typename T>
void TestTensorOpMultiply(ML::DataType dataType, ML::Backend* backend)
{
    ML::Device* device = backend->GetDevice(0);
    ML::SetCurrentDevice(device);
    if (!device || !device->IsDataTypeSupported(dataType))
    {
        ML_LOG(info, "{}.Multiply({}): not supported!", backend->m_name, ML::GetDataTypeName(dataType));
        return;
    }

    ML_LOG(info, "{}.Multiply({}): start", backend->m_name, ML::GetDataTypeName(dataType));

    static_assert(rad::is_floating_point_v<T> || std::is_integral_v<T>);
    using ComputeType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    a.FillConstant(ComputeType(2));
    b.FillConstant(ComputeType(2));
    ML::Tensor c = a.Multiply(b);

    c.MultiplyScalarInPlace(ComputeType(2));

    std::vector<uint8_t> dataBuffer;
    dataBuffer.resize(c.GetDataSize());
    c.Read(dataBuffer.data(), 0, dataBuffer.size());
    const T* results = reinterpret_cast<const T*>(dataBuffer.data());
    for (size_t i = 0; i < c.GetElementCount(); ++i)
    {
        ASSERT_EQ(results[i], 8);
    }
}

template <typename T>
void TestTensorOpDivide(ML::DataType dataType, ML::Backend* backend)
{
    ML::Device* device = backend->GetDevice(0);
    ML::SetCurrentDevice(device);
    if (!device || !device->IsDataTypeSupported(dataType))
    {
        ML_LOG(info, "{}.Divide({}): not supported!", backend->m_name, ML::GetDataTypeName(dataType));
        return;
    }

    ML_LOG(info, "{}.Divide({}): start", backend->m_name, ML::GetDataTypeName(dataType));
    static_assert(rad::is_floating_point_v<T> || std::is_integral_v<T>);
    using ComputeType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    a.FillConstant(ComputeType(8));
    b.FillConstant(ComputeType(2));
    ML::Tensor c = a.Divide(b);

    c.DivideScalarInPlace(ComputeType(2));

    std::vector<uint8_t> dataBuffer;
    dataBuffer.resize(c.GetDataSize());
    c.Read(dataBuffer.data(), 0, dataBuffer.size());
    const T* results = reinterpret_cast<const T*>(dataBuffer.data());
    for (size_t i = 0; i < c.GetElementCount(); ++i)
    {
        ASSERT_EQ(results[i], 2);
    }
}

TEST(TensorOp, Add)
{
    for (auto backend : g_backends)
    {
        TestTensorOpAdd<rad::Float32>(ML::DataType::Float32, backend);
        TestTensorOpAdd<rad::Float16>(ML::DataType::Float16, backend);
        TestTensorOpAdd<rad::BFloat16>(ML::DataType::BFloat16, backend);
        TestTensorOpAdd<rad::Float8E4M3>(ML::DataType::Float8E4M3, backend);
        TestTensorOpAdd<rad::Float8E5M2>(ML::DataType::Float8E5M2, backend);
        TestTensorOpAdd<rad::Sint8>(ML::DataType::Sint8, backend);
        TestTensorOpAdd<rad::Sint16>(ML::DataType::Sint16, backend);
        TestTensorOpAdd<rad::Sint32>(ML::DataType::Sint32, backend);
        TestTensorOpAdd<rad::Sint64>(ML::DataType::Sint64, backend);
        TestTensorOpAdd<rad::Uint8>(ML::DataType::Uint8, backend);
        TestTensorOpAdd<rad::Uint16>(ML::DataType::Uint16, backend);
        TestTensorOpAdd<rad::Uint32>(ML::DataType::Uint32, backend);
        TestTensorOpAdd<rad::Uint64>(ML::DataType::Uint64, backend);
    }
}

TEST(TensorOp, Subtract)
{
    for (auto backend : g_backends)
    {
        TestTensorOpSubtract<rad::Float32>(ML::DataType::Float32, backend);
        TestTensorOpSubtract<rad::Float16>(ML::DataType::Float16, backend);
        TestTensorOpSubtract<rad::BFloat16>(ML::DataType::BFloat16, backend);
        TestTensorOpSubtract<rad::Float8E4M3>(ML::DataType::Float8E4M3, backend);
        TestTensorOpSubtract<rad::Float8E5M2>(ML::DataType::Float8E5M2, backend);
        TestTensorOpSubtract<rad::Sint8>(ML::DataType::Sint8, backend);
        TestTensorOpSubtract<rad::Sint16>(ML::DataType::Sint16, backend);
        TestTensorOpSubtract<rad::Sint32>(ML::DataType::Sint32, backend);
        TestTensorOpSubtract<rad::Sint64>(ML::DataType::Sint64, backend);
        TestTensorOpSubtract<rad::Uint8>(ML::DataType::Uint8, backend);
        TestTensorOpSubtract<rad::Uint16>(ML::DataType::Uint16, backend);
        TestTensorOpSubtract<rad::Uint32>(ML::DataType::Uint32, backend);
        TestTensorOpSubtract<rad::Uint64>(ML::DataType::Uint64, backend);
    }
}

TEST(TensorOp, Multiply)
{
    for (auto backend : g_backends)
    {
        TestTensorOpMultiply<rad::Float32>(ML::DataType::Float32, backend);
        TestTensorOpMultiply<rad::Float16>(ML::DataType::Float16, backend);
        TestTensorOpMultiply<rad::BFloat16>(ML::DataType::BFloat16, backend);
        TestTensorOpMultiply<rad::Float8E4M3>(ML::DataType::Float8E4M3, backend);
        TestTensorOpMultiply<rad::Float8E5M2>(ML::DataType::Float8E5M2, backend);
        TestTensorOpMultiply<rad::Sint8>(ML::DataType::Sint8, backend);
        TestTensorOpMultiply<rad::Sint16>(ML::DataType::Sint16, backend);
        TestTensorOpMultiply<rad::Sint32>(ML::DataType::Sint32, backend);
        TestTensorOpMultiply<rad::Sint64>(ML::DataType::Sint64, backend);
        TestTensorOpMultiply<rad::Uint8>(ML::DataType::Uint8, backend);
        TestTensorOpMultiply<rad::Uint16>(ML::DataType::Uint16, backend);
        TestTensorOpMultiply<rad::Uint32>(ML::DataType::Uint32, backend);
        TestTensorOpMultiply<rad::Uint64>(ML::DataType::Uint64, backend);
    }
}

TEST(TensorOp, Divide)
{
    for (auto backend : g_backends)
    {
        TestTensorOpDivide<rad::Float32>(ML::DataType::Float32, backend);
        TestTensorOpDivide<rad::Float16>(ML::DataType::Float16, backend);
        TestTensorOpDivide<rad::BFloat16>(ML::DataType::BFloat16, backend);
        TestTensorOpDivide<rad::Float8E4M3>(ML::DataType::Float8E4M3, backend);
        TestTensorOpDivide<rad::Float8E5M2>(ML::DataType::Float8E5M2, backend);
        TestTensorOpDivide<rad::Sint8>(ML::DataType::Sint8, backend);
        TestTensorOpDivide<rad::Sint16>(ML::DataType::Sint16, backend);
        TestTensorOpDivide<rad::Sint32>(ML::DataType::Sint32, backend);
        TestTensorOpDivide<rad::Sint64>(ML::DataType::Sint64, backend);
        TestTensorOpDivide<rad::Uint8>(ML::DataType::Uint8, backend);
        TestTensorOpDivide<rad::Uint16>(ML::DataType::Uint16, backend);
        TestTensorOpDivide<rad::Uint32>(ML::DataType::Uint32, backend);
        TestTensorOpDivide<rad::Uint64>(ML::DataType::Uint64, backend);
    }
}
