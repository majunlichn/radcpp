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

bool VerifyDimByDim(const std::vector<size_t> sizes, const std::vector<size_t>& strides,
    const std::function<bool(rad::ArrayRef<size_t> indices)> verify,
    std::vector<size_t>& indices, size_t dimIndex, size_t parallelism)
{
    if (dimIndex == sizes.size() - 1)
    {
        // Iterate the last dimension:
        for (size_t i = 0; i < sizes[dimIndex]; ++i)
        {
            indices[dimIndex] = i;
            size_t index = std::inner_product(indices.begin(), indices.end(), strides.begin(), size_t(0));
            if (!verify(indices))
            {
                return false;
            }
        }
        return true;
    }
    else
    {
        const size_t minParallelism = std::min<size_t>(2, std::thread::hardware_concurrency() / 2);
        if (parallelism >= minParallelism)
        {
            tf::Executor executor;
            std::atomic<bool> success(true);
            // Iterate recursively:
            for (size_t i = 0; i < sizes[dimIndex]; ++i)
            {
                executor.silent_async([&, indices, i]() mutable {
                    indices[dimIndex] = i;
                    if (!VerifyDimByDim(sizes, strides, verify, indices, dimIndex + 1, 0))
                    {
                        success = false;
                    }
                    });
            }
            executor.wait_for_all();
            return success;
        }
        else
        {
            // Iterate recursively:
            for (size_t i = 0; i < sizes[dimIndex]; ++i)
            {
                indices[dimIndex] = i;
                if (!VerifyDimByDim(sizes, strides, verify, indices, dimIndex + 1, parallelism * sizes[dimIndex + 1]))
                {
                    return false;
                }
            }
            return true;
        }
    }
}

bool Verify(const std::vector<size_t>& sizes, const std::vector<size_t>& strides,
    const std::function<bool(rad::ArrayRef<size_t> indices)>& verify)
{
    std::vector<size_t> indices(sizes.size(), 0);
    rad::Stopwatch stopwatch;
    stopwatch.Start();
    bool result = VerifyDimByDim(sizes, strides, verify, indices, size_t(0), sizes[0]);
    stopwatch.Stop();
    VKPP_LOG(info, "Verification completed in {:.3f} ms", stopwatch.GetElapsedMilliseconds());
    return result;
}

void TestElementWiseSqrt(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides = {})
{
    VKPP_LOG(info, "ElementWiseSqrt: sizes={}; strides={};", rad::ToString(sizes), rad::ToString(strides));
    rad::Ref<vkpp::Tensor> tensor = RAD_NEW vkpp::Tensor(g_device);
    tensor->Init(vk::ComponentTypeKHR::eFloat16, sizes, strides);

    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<uint16_t> inputData = tensor->GenerateData<uint16_t>(
        [&](rad::ArrayRef<size_t> indices) { return rad::fp16_ieee_from_fp32_value(dist(gen)); });
    tensor->Write(inputData.data());

    rad::Ref<vkpp::TensorOpElementWiseUnary> opSqrt = RAD_NEW vkpp::TensorOpElementWiseUnary(g_device);
    vkpp::TensorOpElementWiseUnaryDesc opDesc = {};
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
    const float tolerance = 0.0005f;
    vkpp::TensorIterator iter(tensor->m_sizes);
    iter.ForEach([&](rad::ArrayRef<size_t> indices) {
        size_t index = std::inner_product(indices.begin(), indices.end(), tensor->m_strides.begin(), size_t(0));
        float result = rad::fp16_ieee_to_fp32_value(results[index]);
        float resultRef = std::sqrt(rad::fp16_ieee_to_fp32_value(inputData[index]));
        float diff = std::abs(result - resultRef);
        if (diff > maxDiff)
        {
            maxDiff = diff;
        }
        EXPECT_TRUE(diff < tolerance);
        if (diff >= tolerance)
        {
            VKPP_LOG(err, "Verification failed at {}", rad::ToString(indices));
            return false;
        }
        return true;
        });
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
    TestElementWiseSqrt({ 2, 4, 512, 512 });

    vkpp::TensorIterator iter({ 2, 4, 8, 8 });
    iter.ForEach([](rad::ArrayRef<size_t> indices) {
        VKPP_LOG(info, "Indices: {}", rad::ToString(indices));
        });
}
