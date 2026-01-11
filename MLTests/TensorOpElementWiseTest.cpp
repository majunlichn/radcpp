#include <MLCore/MLCore.h>

#include <gtest/gtest.h>


template <typename T>
void TestTensorOpAdd(ML::DataType dataType)
{
    ML::Device* device = ML::GetCurrentDevice();
    if (device->IsDataTypeSupported(dataType))
    {
        ML_LOG(info, "TensorOp.Add({}): start", ML::GetDataTypeName(dataType));
    }
    else
    {
        ML_LOG(info, "TensorOp.Add({}): not supported!", ML::GetDataTypeName(dataType));
        return;
    }

    static_assert(rad::is_floating_point_v<T> || std::is_integral_v<T>);
    using ComputeType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    a.Fill(ComputeType(1));
    b.Fill(ComputeType(1));
    ML::Tensor c = a.Add(b, ComputeType(2));

    c += ComputeType(1);

    std::vector<uint8_t> dataBuffer;
    dataBuffer.resize(c.GetDataSize());
    c.Read(dataBuffer.data(), 0, dataBuffer.size());
    const T* results = reinterpret_cast<const T*>(dataBuffer.data());
    for (size_t i = 0; i < c.GetElementCount(); ++i)
    {
        ASSERT_EQ(results[i], 4);
    }
}

TEST(TensorOp, Add)
{
    TestTensorOpAdd<rad::Float32>(ML::DataType::Float32);
    TestTensorOpAdd<rad::Float16>(ML::DataType::Float16);
    TestTensorOpAdd<rad::BFloat16>(ML::DataType::BFloat16);
    TestTensorOpAdd<rad::Float8E4M3>(ML::DataType::Float8E4M3);
    TestTensorOpAdd<rad::Float8E5M2>(ML::DataType::Float8E5M2);
    TestTensorOpAdd<rad::Sint8>(ML::DataType::Sint8);
    TestTensorOpAdd<rad::Sint16>(ML::DataType::Sint16);
    TestTensorOpAdd<rad::Sint32>(ML::DataType::Sint32);
    TestTensorOpAdd<rad::Sint64>(ML::DataType::Sint64);
    TestTensorOpAdd<rad::Uint8>(ML::DataType::Uint8);
    TestTensorOpAdd<rad::Uint16>(ML::DataType::Uint16);
    TestTensorOpAdd<rad::Uint32>(ML::DataType::Uint32);
    TestTensorOpAdd<rad::Uint64>(ML::DataType::Uint64);
}

template <typename T>
void TestTensorOpSubtract(ML::DataType dataType)
{
    ML::Device* device = ML::GetCurrentDevice();
    if (device->IsDataTypeSupported(dataType))
    {
        ML_LOG(info, "TensorOp.Subtract({}): start", ML::GetDataTypeName(dataType));
    }
    else
    {
        ML_LOG(info, "TensorOp.Subtract({}): not supported!", ML::GetDataTypeName(dataType));
        return;
    }

    static_assert(rad::is_floating_point_v<T> || std::is_integral_v<T>);
    using ComputeType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    a.Fill(ComputeType(4));
    b.Fill(ComputeType(1));
    ML::Tensor c = a.Subtract(b, ComputeType(2));

    c -= ComputeType(1);

    std::vector<uint8_t> dataBuffer;
    dataBuffer.resize(c.GetDataSize());
    c.Read(dataBuffer.data(), 0, dataBuffer.size());
    const T* results = reinterpret_cast<const T*>(dataBuffer.data());
    for (size_t i = 0; i < c.GetElementCount(); ++i)
    {
        ASSERT_EQ(results[i], 1);
    }
}

TEST(TensorOp, Subtract)
{
    TestTensorOpSubtract<rad::Float32>(ML::DataType::Float32);
    TestTensorOpSubtract<rad::Float16>(ML::DataType::Float16);
    TestTensorOpSubtract<rad::BFloat16>(ML::DataType::BFloat16);
    TestTensorOpSubtract<rad::Float8E4M3>(ML::DataType::Float8E4M3);
    TestTensorOpSubtract<rad::Float8E5M2>(ML::DataType::Float8E5M2);
    TestTensorOpSubtract<rad::Sint8>(ML::DataType::Sint8);
    TestTensorOpSubtract<rad::Sint16>(ML::DataType::Sint16);
    TestTensorOpSubtract<rad::Sint32>(ML::DataType::Sint32);
    TestTensorOpSubtract<rad::Sint64>(ML::DataType::Sint64);
    TestTensorOpSubtract<rad::Uint8>(ML::DataType::Uint8);
    TestTensorOpSubtract<rad::Uint16>(ML::DataType::Uint16);
    TestTensorOpSubtract<rad::Uint32>(ML::DataType::Uint32);
    TestTensorOpSubtract<rad::Uint64>(ML::DataType::Uint64);
}

template <typename T>
void TestTensorOpMultiply(ML::DataType dataType)
{
    ML::Device* device = ML::GetCurrentDevice();
    if (device->IsDataTypeSupported(dataType))
    {
        ML_LOG(info, "TensorOp.Multiply({}): start", ML::GetDataTypeName(dataType));
    }
    else
    {
        ML_LOG(info, "TensorOp.Multiply({}): not supported!", ML::GetDataTypeName(dataType));
        return;
    }

    static_assert(rad::is_floating_point_v<T> || std::is_integral_v<T>);
    using ComputeType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    a.Fill(ComputeType(2));
    b.Fill(ComputeType(2));
    ML::Tensor c = a * b;

    c *= ComputeType(2);

    std::vector<uint8_t> dataBuffer;
    dataBuffer.resize(c.GetDataSize());
    c.Read(dataBuffer.data(), 0, dataBuffer.size());
    const T* results = reinterpret_cast<const T*>(dataBuffer.data());
    for (size_t i = 0; i < c.GetElementCount(); ++i)
    {
        ASSERT_EQ(results[i], 8);
    }
}

TEST(TensorOp, Multiply)
{
    TestTensorOpMultiply<rad::Float32>(ML::DataType::Float32);
    TestTensorOpMultiply<rad::Float16>(ML::DataType::Float16);
    TestTensorOpMultiply<rad::BFloat16>(ML::DataType::BFloat16);
    TestTensorOpMultiply<rad::Float8E4M3>(ML::DataType::Float8E4M3);
    TestTensorOpMultiply<rad::Float8E5M2>(ML::DataType::Float8E5M2);
    TestTensorOpMultiply<rad::Sint8>(ML::DataType::Sint8);
    TestTensorOpMultiply<rad::Sint16>(ML::DataType::Sint16);
    TestTensorOpMultiply<rad::Sint32>(ML::DataType::Sint32);
    TestTensorOpMultiply<rad::Sint64>(ML::DataType::Sint64);
    TestTensorOpMultiply<rad::Uint8>(ML::DataType::Uint8);
    TestTensorOpMultiply<rad::Uint16>(ML::DataType::Uint16);
    TestTensorOpMultiply<rad::Uint32>(ML::DataType::Uint32);
    TestTensorOpMultiply<rad::Uint64>(ML::DataType::Uint64);
}

template <typename T>
void TestTensorOpDivide(ML::DataType dataType)
{
    ML::Device* device = ML::GetCurrentDevice();
    if (device->IsDataTypeSupported(dataType))
    {
        ML_LOG(info, "TensorOp.Divide({}): start", ML::GetDataTypeName(dataType));
    }
    else
    {
        ML_LOG(info, "TensorOp.Divide({}): not supported!", ML::GetDataTypeName(dataType));
        return;
    }

    static_assert(rad::is_floating_point_v<T> || std::is_integral_v<T>);
    using ComputeType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    a.Fill(ComputeType(8));
    b.Fill(ComputeType(2));
    ML::Tensor c = a / b;

    c /= ComputeType(2);

    std::vector<uint8_t> dataBuffer;
    dataBuffer.resize(c.GetDataSize());
    c.Read(dataBuffer.data(), 0, dataBuffer.size());
    const T* results = reinterpret_cast<const T*>(dataBuffer.data());
    for (size_t i = 0; i < c.GetElementCount(); ++i)
    {
        ASSERT_EQ(results[i], 2);
    }
}

TEST(TensorOp, Divide)
{
    TestTensorOpDivide<rad::Float32>(ML::DataType::Float32);
    TestTensorOpDivide<rad::Float16>(ML::DataType::Float16);
    TestTensorOpDivide<rad::BFloat16>(ML::DataType::BFloat16);
    TestTensorOpDivide<rad::Float8E4M3>(ML::DataType::Float8E4M3);
    TestTensorOpDivide<rad::Float8E5M2>(ML::DataType::Float8E5M2);
    TestTensorOpDivide<rad::Sint8>(ML::DataType::Sint8);
    TestTensorOpDivide<rad::Sint16>(ML::DataType::Sint16);
    TestTensorOpDivide<rad::Sint32>(ML::DataType::Sint32);
    TestTensorOpDivide<rad::Sint64>(ML::DataType::Sint64);
    TestTensorOpDivide<rad::Uint8>(ML::DataType::Uint8);
    TestTensorOpDivide<rad::Uint16>(ML::DataType::Uint16);
    TestTensorOpDivide<rad::Uint32>(ML::DataType::Uint32);
    TestTensorOpDivide<rad::Uint64>(ML::DataType::Uint64);
}

template <typename T>
void TestTensorOpRemainder(ML::DataType dataType)
{
    ML::Device* device = ML::GetCurrentDevice();
    if (device->IsDataTypeSupported(dataType))
    {
        ML_LOG(info, "TensorOp.Remainder({}): start", ML::GetDataTypeName(dataType));
    }
    else
    {
        ML_LOG(info, "TensorOp.Remainder({}): not supported!", ML::GetDataTypeName(dataType));
        return;
    }

    static_assert(rad::is_floating_point_v<T> || std::is_integral_v<T>);
    using ComputeType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    ComputeType dividend = ComputeType(7);
    ComputeType divisor = ComputeType(3);
    ComputeType ref = ComputeType(1);
    if (rad::is_signed_v<T>)
    {
        dividend = ComputeType(-7);
        divisor = ComputeType(3);
        ref = ComputeType(2);
    }
    a.Fill(dividend);
    b.Fill(divisor);
    ML::Tensor c = a % b;

    std::vector<uint8_t> dataBuffer;
    dataBuffer.resize(c.GetDataSize());
    c.Read(dataBuffer.data(), 0, dataBuffer.size());
    const T* results = reinterpret_cast<const T*>(dataBuffer.data());
    for (size_t i = 0; i < c.GetElementCount(); ++i)
    {
        ASSERT_EQ(results[i], ref);
    }
}

TEST(TensorOp, Remainder)
{
    TestTensorOpRemainder<rad::Float32>(ML::DataType::Float32);
    TestTensorOpRemainder<rad::Float16>(ML::DataType::Float16);
    TestTensorOpRemainder<rad::BFloat16>(ML::DataType::BFloat16);
    TestTensorOpRemainder<rad::Float8E4M3>(ML::DataType::Float8E4M3);
    TestTensorOpRemainder<rad::Float8E5M2>(ML::DataType::Float8E5M2);
    TestTensorOpRemainder<rad::Sint8>(ML::DataType::Sint8);
    TestTensorOpRemainder<rad::Sint16>(ML::DataType::Sint16);
    TestTensorOpRemainder<rad::Sint32>(ML::DataType::Sint32);
    TestTensorOpRemainder<rad::Sint64>(ML::DataType::Sint64);
    TestTensorOpRemainder<rad::Uint8>(ML::DataType::Uint8);
    TestTensorOpRemainder<rad::Uint16>(ML::DataType::Uint16);
    TestTensorOpRemainder<rad::Uint32>(ML::DataType::Uint32);
    TestTensorOpRemainder<rad::Uint64>(ML::DataType::Uint64);
}
