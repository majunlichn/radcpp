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
    using ScalarType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    a.Fill(ScalarType(1));
    b.Fill(ScalarType(1));
    ML::Tensor c = a.Add(b, ScalarType(2));

    c += ScalarType(1);

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
    TestTensorOpAdd<rad::Float16>(ML::DataType::Float16);
    TestTensorOpAdd<rad::Float32>(ML::DataType::Float32);
    TestTensorOpAdd<rad::Float64>(ML::DataType::Float64);
    TestTensorOpAdd<rad::Sint8>(ML::DataType::Sint8);
    TestTensorOpAdd<rad::Sint16>(ML::DataType::Sint16);
    TestTensorOpAdd<rad::Sint32>(ML::DataType::Sint32);
    TestTensorOpAdd<rad::Sint64>(ML::DataType::Sint64);
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
    using ScalarType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    a.Fill(ScalarType(4));
    b.Fill(ScalarType(1));
    ML::Tensor c = a.Subtract(b, ScalarType(2));

    c -= ScalarType(1);

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
    TestTensorOpSubtract<rad::Float16>(ML::DataType::Float16);
    TestTensorOpSubtract<rad::Float32>(ML::DataType::Float32);
    TestTensorOpSubtract<rad::Float64>(ML::DataType::Float64);
    TestTensorOpSubtract<rad::Sint8>(ML::DataType::Sint8);
    TestTensorOpSubtract<rad::Sint16>(ML::DataType::Sint16);
    TestTensorOpSubtract<rad::Sint32>(ML::DataType::Sint32);
    TestTensorOpSubtract<rad::Sint64>(ML::DataType::Sint64);
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
    using ScalarType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    a.Fill(ScalarType(2));
    b.Fill(ScalarType(2));
    ML::Tensor c = a * b;

    c *= ScalarType(2);

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
    TestTensorOpMultiply<rad::Float16>(ML::DataType::Float16);
    TestTensorOpMultiply<rad::Float32>(ML::DataType::Float32);
    TestTensorOpMultiply<rad::Float64>(ML::DataType::Float64);
    TestTensorOpMultiply<rad::Sint8>(ML::DataType::Sint8);
    TestTensorOpMultiply<rad::Sint16>(ML::DataType::Sint16);
    TestTensorOpMultiply<rad::Sint32>(ML::DataType::Sint32);
    TestTensorOpMultiply<rad::Sint64>(ML::DataType::Sint64);
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
    using ScalarType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    a.Fill(ScalarType(8));
    b.Fill(ScalarType(2));
    ML::Tensor c = a / b;

    c /= ScalarType(2);

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
    TestTensorOpDivide<rad::Float16>(ML::DataType::Float16);
    TestTensorOpDivide<rad::Float32>(ML::DataType::Float32);
    TestTensorOpDivide<rad::Float64>(ML::DataType::Float64);
    TestTensorOpDivide<rad::Sint8>(ML::DataType::Sint8);
    TestTensorOpDivide<rad::Sint16>(ML::DataType::Sint16);
    TestTensorOpDivide<rad::Sint32>(ML::DataType::Sint32);
    TestTensorOpDivide<rad::Sint64>(ML::DataType::Sint64);
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
    using ScalarType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    ScalarType dividend = ScalarType(7);
    ScalarType divisor = ScalarType(3);
    ScalarType ref = ScalarType(1);
    if (rad::is_signed_v<T>)
    {
        dividend = ScalarType(-7);
        divisor = ScalarType(3);
        ref = ScalarType(2);
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
    TestTensorOpRemainder<rad::Float16>(ML::DataType::Float16);
    TestTensorOpRemainder<rad::Float32>(ML::DataType::Float32);
    TestTensorOpRemainder<rad::Float64>(ML::DataType::Float64);
    TestTensorOpRemainder<rad::Sint8>(ML::DataType::Sint8);
    TestTensorOpRemainder<rad::Sint16>(ML::DataType::Sint16);
    TestTensorOpRemainder<rad::Sint32>(ML::DataType::Sint32);
    TestTensorOpRemainder<rad::Sint64>(ML::DataType::Sint64);
}

template <typename T>
void TestTensorOpBitwise(ML::DataType dataType)
{
    ML::Device* device = ML::GetCurrentDevice();
    if (device->IsDataTypeSupported(dataType))
    {
        ML_LOG(info, "TensorOp.Bitwise({}): start", ML::GetDataTypeName(dataType));
    }
    else
    {
        ML_LOG(info, "TensorOp.Bitwise({}): not supported!", ML::GetDataTypeName(dataType));
        return;
    }

    static_assert(std::is_integral_v<T>);

    ML::Tensor a = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);
    ML::Tensor b = ML::MakeTensor({ 2, 4, 32, 32 }, dataType);

    a.Fill(0b1010);
    b.Fill(0b0110);
    ML::Tensor c = a & b;

    std::vector<uint8_t> dataBuffer;
    dataBuffer.resize(c.GetDataSize());
    c.Read(dataBuffer.data(), 0, dataBuffer.size());
    const T* results = reinterpret_cast<const T*>(dataBuffer.data());
    for (size_t i = 0; i < c.GetElementCount(); ++i)
    {
        ASSERT_EQ(results[i], 0b0010);
    }

    c = a | b;
    c.Read(dataBuffer.data(), 0, dataBuffer.size());
    results = reinterpret_cast<const T*>(dataBuffer.data());
    for (size_t i = 0; i < c.GetElementCount(); ++i)
    {
        ASSERT_EQ(results[i], 0b1110);
    }

    c = a ^ b;
    c.Read(dataBuffer.data(), 0, dataBuffer.size());
    results = reinterpret_cast<const T*>(dataBuffer.data());
    for (size_t i = 0; i < c.GetElementCount(); ++i)
    {
        ASSERT_EQ(results[i], 0b1100);
    }
}

TEST(TensorOp, Bitwise)
{
    TestTensorOpBitwise<rad::Sint8>(ML::DataType::Sint8);
    TestTensorOpBitwise<rad::Sint16>(ML::DataType::Sint16);
    TestTensorOpBitwise<rad::Sint32>(ML::DataType::Sint32);
    TestTensorOpBitwise<rad::Sint64>(ML::DataType::Sint64);
    TestTensorOpBitwise<rad::Uint8>(ML::DataType::Uint8);
    TestTensorOpBitwise<rad::Uint16>(ML::DataType::Uint16);
    TestTensorOpBitwise<rad::Uint32>(ML::DataType::Uint32);
    TestTensorOpBitwise<rad::Uint64>(ML::DataType::Uint64);
}
