#include <radcpp/GPU/VulkanInstance.h>
#include <radcpp/GPU/VulkanDevice.h>
#include <radcpp/GPU/VulkanBuffer.h>
#include <radcpp/GPU/VulkanImage.h>
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

        vk::Format colorFormat = device->FindFormat({ vk::Format::eR8G8B8A8Unorm },
            {}, { vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eColorAttachment }, {});

        rad::Ref<rad::VulkanImage> colorImage;
        if (colorFormat != vk::Format::eUndefined)
        {
            LOG_VULKAN(info, "Color format: {}", vk::to_string(colorFormat));
            vk::ImageCreateInfo imageInfo;
            imageInfo.imageType = vk::ImageType::e2D;
            imageInfo.format = colorFormat;
            imageInfo.extent.width = 1920;
            imageInfo.extent.height = 1080;
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = vk::SampleCountFlagBits::e1;
            imageInfo.tiling = vk::ImageTiling::eOptimal;
            imageInfo.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment;
            imageInfo.initialLayout = vk::ImageLayout::eUndefined;
            VmaAllocationCreateInfo allocCreateInfo = {};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
            colorImage = RAD_NEW rad::VulkanImage(device, imageInfo, allocCreateInfo);
        }

        vk::Format dsFormat = device->FindFormat({ vk::Format::eD24UnormS8Uint, vk::Format::eD32SfloatS8Uint },
            {}, { vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eDepthStencilAttachment }, {});
        rad::Ref<rad::VulkanImage> dsImage;
        if (dsFormat != vk::Format::eUndefined)
        {
            LOG_VULKAN(info, "DepthStencil format: {}", vk::to_string(dsFormat));
            vk::ImageCreateInfo imageInfo;
            imageInfo.imageType = vk::ImageType::e2D;
            imageInfo.format = dsFormat;
            imageInfo.extent.width = 1920;
            imageInfo.extent.height = 1080;
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = vk::SampleCountFlagBits::e1;
            imageInfo.tiling = vk::ImageTiling::eOptimal;
            imageInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
            imageInfo.initialLayout = vk::ImageLayout::eUndefined;
            VmaAllocationCreateInfo allocCreateInfo = {};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
            dsImage = RAD_NEW rad::VulkanImage(device, imageInfo, allocCreateInfo);
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
