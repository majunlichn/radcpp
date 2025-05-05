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

TEST(Tensor, ElementWise)
{
    rad::Ref<vkpp::CommandPool> cmdPool = g_device->CreateCommandPool(
        vkpp::QueueFamily::Universal,
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    vk::raii::CommandBuffers cmdBuffers = cmdPool->AllocatePrimary(1);

    rad::Ref<vkpp::DescriptorPool> descPool = g_device->CreateDescriptorPool(1,
        {   // type, count
            { vk::DescriptorType::eStorageBuffer, 1 }
        });
    vk::raii::DescriptorSetLayout descSetLayout = g_device->CreateDescriptorSetLayout(
        {   // binding, type, count, stageFlags, pImmutableSamplers
            { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
        });
    vk::raii::DescriptorSets descSets = descPool->Allocate({ descSetLayout });

    rad::Ref<vkpp::Tensor> tensor = RAD_NEW vkpp::Tensor(g_device);
    tensor->Init(vk::ComponentTypeKHR::eFloat16, { 1, 4, 1024, 1024 }, vkpp::Tensor::MemoryLayout::NCHW);
    tensor->FillConstant<uint16_t>(0);

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<uint16_t> initData = tensor->GenerateBufferData<uint16_t>(
        [&](size_t index, std::initializer_list<size_t> coord) { return rad::fp16_ieee_from_fp32_value(dist(eng)); });
    tensor->Write(initData.data());

    rad::Ref<vkpp::TensorOpElementWiseUnary> op = RAD_NEW vkpp::TensorOpElementWiseUnary(g_device);
    vkpp::TensorOpElementWiseUnaryDesc opDesc;
    opDesc.opName = "sqrt";
    opDesc.dataType = vk::ComponentTypeKHR::eFloat16;
    opDesc.sizes = tensor->m_sizes;
    opDesc.inputStrides = tensor->m_strides;
    opDesc.outputStrides = tensor->m_strides;
    op->Init(opDesc);
    op->SetTensor(1, tensor.get());
    op->SetTensor(2, tensor.get());

    glm::uvec3 groupCount = {};
    groupCount.x = rad::DivRoundUp<uint32_t>(static_cast<uint32_t>(tensor->m_sizes[3]), 16u);    // W
    groupCount.y = rad::DivRoundUp<uint32_t>(static_cast<uint32_t>(tensor->m_sizes[2]), 16u);    // H
    groupCount.z = static_cast<uint32_t>(tensor->m_sizes[1]);   // C
    op->Execute(groupCount);

    // Check the results:
    std::vector<uint16_t> results(tensor->GetBufferElementCount());
    tensor->Read(results.data());
    for (size_t i = 0; i < results.size(); ++i)
    {
        float result = rad::fp16_ieee_to_fp32_value(results[i]);
        float refValue = std::sqrt(rad::fp16_ieee_to_fp32_value(initData[i]));
        float diff = std::abs(result - refValue);
        EXPECT_TRUE(diff < 0.001f);
        if (diff >= 0.001f)
        {
            VKPP_LOG(err, "Verification failed: Result={}; Ref={}; Diff={:.6f};",
                result, refValue, std::abs(result - refValue));
            break;
        }
    }
}
