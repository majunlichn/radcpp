#include <rad/ML/MLDevice.h>
#include <rad/ML/MLContext.h>
#include <rad/ML/MLTensor.h>
#include <rad/ML/MLCpuDevice.h>
#include <rad/ML/MLCpuTensor.h>

#include <rad/IO/Logging.h>

#include <gtest/gtest.h>

template <typename T>
void TestMLTensorAdd(rad::MLDataType dataType)
{
    static_assert(rad::is_floating_point_v<T> || std::is_integral_v<T>);
    using Alpha = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    rad::Ref<rad::MLCpuDevice> device = RAD_NEW rad::MLCpuDevice();
    rad::Ref<rad::MLContext> context = device->CreateContext();
    rad::Ref<rad::MLTensor> a = device->CreateTensor(dataType, { 2, 4, 8, 8 }, {});
    rad::Ref<rad::MLTensor> b = device->CreateTensor(dataType, { 2, 4, 8, 8 }, {});
    rad::Ref<rad::MLTensor> c = device->CreateTensor(dataType, { 2, 4, 8, 8 }, {});

    context->FillConstant(a.get(), Alpha(1));
    context->FillConstant(b.get(), Alpha(2));
    context->Add(a.get(), b.get(), Alpha(1), c.get());

    const T* results = static_cast<const T*>(c->GetData());
    for (size_t i = 0; i < c->GetElementCount(); ++i)
    {
        EXPECT_EQ(results[i], 3);
    }

    if constexpr (std::is_same_v<T, rad::BFloat16>)
    {
        RAD_LOG(info, "TensorAdd (BFloat16):\n{}", c->ToString());
    }
}

TEST(ML, MLTensorAdd)
{
    TestMLTensorAdd<rad::Float32>(rad::MLDataType::Float32);
    TestMLTensorAdd<rad::Float16>(rad::MLDataType::Float16);
    TestMLTensorAdd<rad::BFloat16>(rad::MLDataType::BFloat16);
    TestMLTensorAdd<rad::Float8E4M3>(rad::MLDataType::Float8E4M3);
    TestMLTensorAdd<rad::Float8E5M2>(rad::MLDataType::Float8E5M2);
    TestMLTensorAdd<rad::Sint8>(rad::MLDataType::Sint8);
    TestMLTensorAdd<rad::Sint16>(rad::MLDataType::Sint16);
    TestMLTensorAdd<rad::Sint32>(rad::MLDataType::Sint32);
    TestMLTensorAdd<rad::Sint64>(rad::MLDataType::Sint64);
    TestMLTensorAdd<rad::Uint8>(rad::MLDataType::Uint8);
    TestMLTensorAdd<rad::Uint16>(rad::MLDataType::Uint16);
    TestMLTensorAdd<rad::Uint32>(rad::MLDataType::Uint32);
    TestMLTensorAdd<rad::Uint64>(rad::MLDataType::Uint64);
}
