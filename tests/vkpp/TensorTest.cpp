#include <vkpp/Core/Instance.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Command.h>
#include <vkpp/Core/Descriptor.h>

#include <vkpp/Compute/Tensor.h>
#include <vkpp/Compute/TensorOp.h>

#include <rad/Core/Float.h>
#include <random>

#include <gtest/gtest.h>

extern rad::Ref<vkpp::Device> g_device;

void TestElementWiseSqrt(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides = {})
{
    rad::Ref<vkpp::Tensor> tensor = RAD_NEW vkpp::Tensor(g_device);
    tensor->Init(vk::ComponentTypeKHR::eFloat16, sizes, strides);

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<uint16_t> inputData = tensor->GenerateData<uint16_t>(
        [&](std::initializer_list<size_t> coord) { return rad::fp16_ieee_from_fp32_value(dist(eng)); });
    tensor->Write(inputData.data());

    rad::Ref<vkpp::TensorOpElementWiseUnary> opSqrt = RAD_NEW vkpp::TensorOpElementWiseUnary(g_device);
    vkpp::TensorOpElementWiseUnaryDesc opDesc;
    opDesc.opName = "sqrt";
    opDesc.dataType = vk::ComponentTypeKHR::eFloat16;
    opDesc.sizes = tensor->m_sizes;
    opDesc.inputStrides = tensor->m_strides;
    opDesc.outputStrides = tensor->m_strides;
    opSqrt->Init(opDesc);
    opSqrt->SetTensor(1, tensor.get());
    opSqrt->SetTensor(2, tensor.get());

    opSqrt->Execute();

    // Check the results:
    std::vector<uint16_t> results(tensor->GetBufferSizeInElements());
    tensor->Read(results.data());
    for (size_t i = 0; i < results.size(); ++i)
    {
        float result = rad::fp16_ieee_to_fp32_value(results[i]);
        float refValue = std::sqrt(rad::fp16_ieee_to_fp32_value(inputData[i]));
        float diff = std::abs(result - refValue);
        EXPECT_TRUE(diff < 0.001f);
        if (diff >= 0.001f)
        {
            VKPP_LOG(err, "Verification failed at buffer index#{}: Result={}; Ref={}; Diff={:.6f};",
                i, result, refValue, std::abs(result - refValue));
            break;
        }
    }
    tensor.reset();
}

TEST(Tensor, ElementWise)
{
    TestElementWiseSqrt({ 2, 2, 10, 4, 120, 120 }, vkpp::Tensor::MakeStrides({ 2, 2, 10, 4, 128, 128 }));
}
