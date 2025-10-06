#include <vkpp/Core/Instance.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Command.h>
#include <vkpp/Core/Descriptor.h>

#include <vkpp/Compute/Tensor.h>
#include <vkpp/Compute/TensorOp.h>

#include <random>

#include <gtest/gtest.h>

extern rad::Ref<vkpp::Device> g_device;

void TestElementWiseSqrt(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides = {})
{
    VKPP_LOG(info, "ElementWiseSqrt: sizes=[{}]; strides=[{}];",
        rad::ToString(sizes, ", "), rad::ToString(strides, ", "));
    rad::Ref<vkpp::Tensor> tensor = RAD_NEW vkpp::Tensor(g_device);
    tensor->Init(vk::ComponentTypeKHR::eFloat16, sizes, strides);

    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<uint16_t> inputData = tensor->GenerateData<uint16_t>(
        [&](rad::ArrayRef<size_t> coords) { return rad::fp16_ieee_from_fp32_value(dist(gen)); });
    tensor->Write(inputData.data());

    rad::Ref<vkpp::TensorElementWiseUnaryOp> opSqrt = RAD_NEW vkpp::TensorElementWiseUnaryOp(g_device);
    vkpp::TensorElementWiseUnaryOpDesc opDesc = {};
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
    std::atomic<float> maxDiff = 0.0f;
    const float tolerance = 0.0005f;
    vkpp::TensorIterator iter(tensor->m_sizes);

    iter.ForEachParallel([&](rad::ArrayRef<size_t> coords) {
        size_t index = std::inner_product(coords.begin(), coords.end(), tensor->m_strides.begin(), size_t(0));
        float result = rad::fp16_ieee_to_fp32_value(results[index]);
        float resultRef = std::sqrt(rad::fp16_ieee_to_fp32_value(inputData[index]));
        float diff = std::abs(result - resultRef);
        if (diff > maxDiff)
        {
            maxDiff = diff;
        }
        });
    EXPECT_TRUE(maxDiff < tolerance);
    if (maxDiff < tolerance)
    {
        VKPP_LOG(info, "Verification passed with MaxDiff={:.6f}", maxDiff);
    }

    tensor.reset();
}

TEST(Tensor, ElementWise)
{
    rad::Ref<vkpp::Tensor> tensor = RAD_NEW vkpp::Tensor(g_device);
    tensor->Init(vk::ComponentTypeKHR::eFloat16, { 1, 4, 30, 30 }, vkpp::Tensor::MakeStrides({ 1, 4, 32, 32 }));
    tensor->FillNormalDistribution();
    tensor->m_sizes = { 1, 4, 32, 32 };
    tensor->DumpTextToFile("TensorPadded.txt");

    TestElementWiseSqrt({ 2, 4, 256, 256 });

    vkpp::TensorIterator iter({ 2, 4, 8, 8 });
    iter.m_permutation = { 1, 3, 2, 0 };
    iter.ForEach([](rad::ArrayRef<size_t> coords) {
        VKPP_LOG(info, "Coords=[{}]", rad::ToString(coords));
        });

    vkpp::HostTensor<float> hostTensor({ 2, 4, 8, 8 },
        vkpp::MakeTensorStrides({ 2, 4, 8, 8 }, { 1, 3, 2, 0 }));
}
