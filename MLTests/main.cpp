#include <rad/System/Application.h>

#include <MLCore/Global.h>
#include <MLCore/CPU/CpuBackend.h>
#include <MLCore/Vulkan/VulkanBackend.h>

#include <gtest/gtest.h>

std::vector<ML::Backend*> g_backends;

int main(int argc, char* argv[])
{
    rad::Application app;
    if (!app.Init(argc, argv))
    {
        std::cerr << "Init failed!" << std::endl;
        return -1;
    }

    ML::Initialize();
    if (ML::Backend* backend = ML::InitCpuBackend("CPU"))
    {
        g_backends.push_back(backend);
    }
    if (ML::Backend* backend = ML::InitVulkanBackend("Vulkan"))
    {
        g_backends.push_back(backend);
    }

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
