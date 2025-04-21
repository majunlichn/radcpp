#include <vkpp/Core/Instance.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Command.h>
#include <vkpp/Core/Descriptor.h>

#include <rad/System/Application.h>

#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
    rad::Application app;
    if (!app.Init(argc, argv))
    {
        std::cerr << "Init failed!" << std::endl;
        return -1;
    }

    rad::Ref<vkpp::Instance> vulkan = RAD_NEW vkpp::Instance();
    vulkan->Init("HelloVulkan", VK_MAKE_VERSION(0, 0, 0));
    if (vulkan->m_physicalDevices.empty())
    {
        VKPP_LOG(err, "No Vulkan device available!");
        return -1;
    }

    rad::Ref<vkpp::Device> device = vulkan->CreateDevice();
    rad::Ref<vkpp::CommandPool> cmdPool = device->CreateCommandPool(
        vkpp::QueueFamily::Graphics,
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    vk::raii::CommandBuffers cmdBuffers = cmdPool->AllocatePrimary(1);

    rad::Ref<vkpp::DescriptorPool> descPool = device->CreateDescriptorPool(1,
        {   // type, count
            { vk::DescriptorType::eStorageBuffer, 1 }
        });
    vk::raii::DescriptorSetLayout descSetLayout = device->CreateDescriptorSetLayout(
        {   // binding, type, count, stageFlags, pImmutableSamplers
            { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
        });
    vk::raii::DescriptorSets descSets = descPool->Allocate({ descSetLayout });

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
