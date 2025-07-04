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
    std::string str = "(";
    for (size_t i = 0; i < dims.size(); ++i)
    {
        str += std::to_string(dims[i]);
        if (i < dims.size() - 1)
        {
            str += ", ";
        }
    }
    str += ")";
    return str;
}

bool VerifyByDimension(const std::vector<size_t> sizes, const std::vector<size_t>& strides,
    const std::function<bool(rad::ArrayRef<size_t> coord)> verify,
    size_t dimIndex, std::vector<size_t>& coord, size_t parallelism)
{
    if (dimIndex == sizes.size() - 1)
    {
        // Iterate the last dimension:
        for (size_t i = 0; i < sizes[dimIndex]; ++i)
        {
            coord[dimIndex] = i;
            size_t index = std::inner_product(coord.begin(), coord.end(), strides.begin(), size_t(0));
            if (!verify(coord))
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
                if (success)
                {
                    executor.silent_async([&, coord, i]() mutable {
                        coord[dimIndex] = i;
                        if (!VerifyByDimension(sizes, strides, verify, dimIndex + 1, coord, 0))
                        {
                            success = false;
                        }
                        });
                }
            }
            executor.wait_for_all();
            return success;
        }
        else
        {
            // Iterate recursively:
            for (size_t i = 0; i < sizes[dimIndex]; ++i)
            {
                coord[dimIndex] = i;
                if (!VerifyByDimension(sizes, strides, verify, dimIndex + 1, coord, parallelism * sizes[dimIndex + 1]))
                {
                    return false;
                }
            }
        }
        return true;
    }
}

bool Verify(const std::vector<size_t>& sizes, const std::vector<size_t>& strides,
    const std::function<bool(rad::ArrayRef<size_t> coord)>& verify)
{
    std::vector<size_t> coord(sizes.size(), 0);
    rad::Stopwatch stopwatch;
    stopwatch.Start();
    bool result = VerifyByDimension(sizes, strides, verify, size_t(0), coord, sizes[0]);
    stopwatch.Stop();
    VKPP_LOG(info, "Verification completed in {:.3f} ms", stopwatch.GetElapsedMilliseconds());
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
        [&](rad::ArrayRef<size_t> coord) { return rad::fp16_ieee_from_fp32_value(dist(gen)); });
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
    float tolerance = 0.0005f;
    Verify(tensor->m_sizes, tensor->m_strides, [&](rad::ArrayRef<size_t> coord) {
        size_t index = std::inner_product(coord.begin(), coord.end(), tensor->m_strides.begin(), size_t(0));
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
            VKPP_LOG(err, "Verification failed at coord {}: Result={}; Ref={}; Diff={:.6f};",
                ToString(coord), result, resultRef, std::abs(result - resultRef));
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
    TestElementWiseSqrt({ 2, 4, 256, 256 });
}
