#include <MLCore/Backend.h>
#include <MLCore/Device.h>
#include <MLCore/Context.h>
#include <MLCore/Tensor.h>
#include <MLCore/Logging.h>
#include <MLCore/Vulkan/VulkanBackend.h>

#include <gtest/gtest.h>

template <typename T>
void TestTensorOpAdd(ML::DataType dataType, ML::Backend* backend)
{
    ML::Device* device = backend->GetDevice(0);
    ML::SetCurrentDevice(device);
    if (!device || !device->IsDataTypeSupported(dataType))
    {
        return;
    }

    ML_LOG(info, "TestTensorOpAdd: {}.{}", backend->m_name, ML::GetDataTypeName(dataType));

    static_assert(rad::is_floating_point_v<T> || std::is_integral_v<T>);
    using ComputeType = std::conditional_t<rad::is_floating_point_v<T>, float, int>;

    rad::Ref<ML::Tensor> a = ML::CreateTensor({ 2, 4, 8, 8 }, dataType);
    rad::Ref<ML::Tensor> b = ML::CreateTensor({ 2, 4, 8, 8 }, dataType);

    a->FillConstant(ComputeType(1));
    b->FillConstant(ComputeType(1));
    auto c = a->Add(b.get(), ComputeType(2));

    c->AddScalarInPlace(ComputeType(1));

    const T* results = static_cast<const T*>(c->GetData());
    std::vector<uint8_t> dataBuffer;
    if (results == nullptr)
    {
        dataBuffer.resize(c->GetDataSize());
        c->Read(dataBuffer.data(), 0 , dataBuffer.size());
        results = reinterpret_cast<const T*>(dataBuffer.data());
    }
    for (size_t i = 0; i < c->GetElementCount(); ++i)
    {
        EXPECT_EQ(results[i], 4);
    }

    if constexpr (std::is_same_v<T, rad::BFloat16> || std::is_same_v<T, rad::Float8E4M3>)
    {
        RAD_LOG(info, "Output:\n{}", c->ToString());
    }
}

TEST(TensorOp, Add)
{
    ML::Initialize();
    ML::InitVulkanBackend();

    std::vector<ML::Backend*> backends;
    if (ML::Backend* backend = ML::GetBackend("CPU"))
    {
        backends.push_back(backend);
    }
    if (ML::Backend* backend = ML::GetBackend("Vulkan"))
    {
        backends.push_back(backend);
    }

    for (auto backend : backends)
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
