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

std::string ToString(rad::ArrayRef<size_t> dims)
{
    std::string result = "(";
    for (size_t i = 0; i < dims.size(); ++i)
    {
        result += std::to_string(dims[i]);
        if (i < dims.size() - 1)
        {
            result += ", ";
        }
    }
    result += ")";
    return result;
}

void TestElementWiseSqrt(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides = {})
{
    VKPP_LOG(info, "ElementWiseSqrt: sizes={}; strides={};", ToString(sizes), ToString(strides));
    rad::Ref<vkpp::Tensor> tensor = RAD_NEW vkpp::Tensor(g_device);
    tensor->Init(vk::ComponentTypeKHR::eFloat16, sizes, strides);

    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<uint16_t> inputData = tensor->GenerateData<uint16_t>(
        [&](std::initializer_list<size_t> coord) { return rad::fp16_ieee_from_fp32_value(dist(gen)); });
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

    // Verification:
    std::vector<uint16_t> results(tensor->GetBufferSizeInElements());
    tensor->Read(results.data());
    float maxDiff = 0.0f;
    for (size_t i = 0; i < results.size(); ++i)
    {
        float result = rad::fp16_ieee_to_fp32_value(results[i]);
        float resultRef = std::sqrt(rad::fp16_ieee_to_fp32_value(inputData[i]));
        float diff = std::abs(result - resultRef);
        if (diff > maxDiff)
        {
            maxDiff = diff;
        }
        EXPECT_TRUE(diff < 0.0005f);
        if (diff >= 0.0005f)
        {
            size_t padDimensionCount = vkpp::Tensor::MaxDimensionCount - tensor->GetDimensionCount();
            VKPP_LOG(err, "Verification failed at buffer#{}: Result={}; Ref={}; Diff={:.6f};",
                i, result, resultRef, std::abs(result - resultRef));
            break;
        }
    }
    if (maxDiff < 0.0005f)
    {
        VKPP_LOG(info, "Verification passed with MaxDiff={:.6f}", maxDiff);
    }

    tensor.reset();
}

TEST(Tensor, ElementWise)
{
    TestElementWiseSqrt({ 2, 2, 10, 4, 120, 120 }, vkpp::Tensor::MakeStrides({ 2, 2, 10, 4, 128, 128 }));
}
