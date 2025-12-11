#include <rad/ML/MLDevice.h>
#include <rad/ML/MLContext.h>
#include <rad/ML/MLTensor.h>
#include <rad/ML/MLCpuDevice.h>
#include <rad/ML/MLCpuTensor.h>

#include <rad/IO/Logging.h>

#include <gtest/gtest.h>

template <typename T>
void TestMLTensorOpAdd(rad::MLDataType dataType)
{
    static_assert(rad::is_floating_point_v<T> || std::is_integral_v<T>);
    using Alpha = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    rad::Ref<rad::MLTensor> a = MLCreateTensor({ 2, 4, 8, 8 }, dataType);
    rad::Ref<rad::MLTensor> b = MLCreateTensor({ 2, 4, 8, 8 }, dataType);

    a->FillConstant(Alpha(1));
    b->FillConstant(Alpha(2));
    auto c = a->Add(b.get());

    const T* results = static_cast<const T*>(c->GetData());
    for (size_t i = 0; i < c->GetElementCount(); ++i)
    {
        EXPECT_EQ(results[i], 3);
    }

    if constexpr (std::is_same_v<T, rad::BFloat16>)
    {
        RAD_LOG(info, "MLTensorContext.Add(2x4x8x8.BFloat16):\n{}", c->ToString());
    }
}

TEST(ML, MLTensorOpAdd)
{
    TestMLTensorOpAdd<rad::Float32>(rad::MLDataType::Float32);
    TestMLTensorOpAdd<rad::Float16>(rad::MLDataType::Float16);
    TestMLTensorOpAdd<rad::BFloat16>(rad::MLDataType::BFloat16);
    TestMLTensorOpAdd<rad::Float8E4M3>(rad::MLDataType::Float8E4M3);
    TestMLTensorOpAdd<rad::Float8E5M2>(rad::MLDataType::Float8E5M2);
    TestMLTensorOpAdd<rad::Sint8>(rad::MLDataType::Sint8);
    TestMLTensorOpAdd<rad::Sint16>(rad::MLDataType::Sint16);
    TestMLTensorOpAdd<rad::Sint32>(rad::MLDataType::Sint32);
    TestMLTensorOpAdd<rad::Sint64>(rad::MLDataType::Sint64);
    TestMLTensorOpAdd<rad::Uint8>(rad::MLDataType::Uint8);
    TestMLTensorOpAdd<rad::Uint16>(rad::MLDataType::Uint16);
    TestMLTensorOpAdd<rad::Uint32>(rad::MLDataType::Uint32);
    TestMLTensorOpAdd<rad::Uint64>(rad::MLDataType::Uint64);
}
