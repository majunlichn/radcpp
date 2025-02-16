#include <radcpp/GPU/VulkanInstance.h>
#include <radcpp/GPU/VulkanDevice.h>
#include <radcpp/IO/Logging.h>

#include <gtest/gtest.h>

TEST(GPU, Vulkan)
{
    try
    {
        rad::Ref<rad::VulkanInstance> vulkan = RAD_NEW rad::VulkanInstance();
        vulkan->Init("VulkanTest", VK_MAKE_VERSION(0, 0, 0));
        if (vulkan->m_physicalDevices.empty())
        {
            LOG_VULKAN(err, "No Vulkan device available!");
            return;
        }
        uint32_t levelCount = 0;
        levelCount = vkpp::GetMaxMipLevel(1024, 2048);
        EXPECT_EQ(levelCount, 12);
        levelCount = vkpp::GetMaxMipLevel(2048, 1024);
        EXPECT_EQ(levelCount, 12);
        levelCount = vkpp::GetMaxMipLevel(64, 128, 256);
        EXPECT_EQ(levelCount, 9);
        levelCount = vkpp::GetMaxMipLevel(256, 128, 64);
        EXPECT_EQ(levelCount, 9);

        rad::Ref<rad::VulkanDevice> device = vulkan->CreateDevice();
        LOG_VULKAN(info, "Device created on {}", device->GetName());
        for (const std::string& extension : device->m_enabledExtensions)
        {
            LOG_VULKAN(info, "Device extension enabled: {}", extension);
        }
    }
    catch (vk::SystemError& err)
    {
        LOG_VULKAN(err, "vk::SystemError: {}", err.what());
        FAIL();
    }
    catch (std::exception& err)
    {
        LOG_VULKAN(err, "std::exception: {}", err.what());
        FAIL();
    }
    catch (...)
    {
        LOG_VULKAN(err, "unknown exception!");
        FAIL();
    }
}
