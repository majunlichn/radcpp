#include <vkpp/Core/Instance.h>
#include <vkpp/Core/Device.h>

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

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
