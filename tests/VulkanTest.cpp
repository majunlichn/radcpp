#include <radcpp/GPU/VulkanInstance.h>
#include <radcpp/GPU/VulkanDevice.h>
#include <radcpp/GPU/VulkanBuffer.h>
#include <radcpp/GPU/VulkanImage.h>
#include <radcpp/GPU/GLSLCompiler.h>
#include <radcpp/IO/File.h>
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
        rad::Ref<rad::VulkanImageView> colorView;
        if (colorFormat != vk::Format::eUndefined)
        {
            LOG_VULKAN(info, "Color format: {}", vk::to_string(colorFormat));
            colorImage = device->CreateImage2DColorAttachment(colorFormat, 1920, 1080);
            colorView = colorImage->CreateView();
        }

        vk::Format dsFormat = device->FindFormat({ vk::Format::eD24UnormS8Uint, vk::Format::eD32SfloatS8Uint },
            {}, { vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eDepthStencilAttachment }, {});
        rad::Ref<rad::VulkanImage> dsImage;
        rad::Ref<rad::VulkanImageView> dsView;
        if (dsFormat != vk::Format::eUndefined)
        {
            LOG_VULKAN(info, "DepthStencil format: {}", vk::to_string(dsFormat));
            dsImage = device->CreateImage2DDepthStencilAttachment(dsFormat, 1920, 1080);
            dsView = dsImage->CreateView();
        }

        rad::GLSLCompiler compiler;
        std::string fragSource = R"(
#version 450 core
#extension GL_EXT_fragment_shader_barycentric  : require
layout(location = 0) out vec4 FragColor;
void FragMain()
{
    FragColor = vec4(gl_BaryCoordEXT, ALPHA);
}
)";
        std::vector<rad::ShaderMacro> macros = {
            { "ALPHA", 1.0f }
        };
        std::vector<uint32_t> fragBinary = compiler.Compile(
            vk::ShaderStageFlagBits::eFragment, "barycentric", fragSource, "FragMain", macros);
        EXPECT_FALSE(fragBinary.empty());
        std::string fragAssembly = compiler.CompileToAssembly(
            vk::ShaderStageFlagBits::eFragment, "barycentric", fragSource, "FragMain", macros);
        rad::File file;
        if (file.Open("barycentric.spv.txt", "w"))
        {
            file.Write(fragAssembly.data(), fragAssembly.size());
            file.Close();
        }
        std::string fragDisassembly = compiler.Disassemble(fragBinary.data(), fragBinary.size());
        if (file.Open("barycentric.dis.txt", "w"))
        {
            file.Write(fragDisassembly.data(), fragDisassembly.size());
            file.Close();
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
