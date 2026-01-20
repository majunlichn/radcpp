#include <rad/System/Application.h>

#include <MLCore/MLCore.h>
#include <argparse/argparse.hpp>

#include <gtest/gtest.h>

struct Options
{
    std::string backend = "CPU";
    int deviceIndex = 0;
} g_options;

int main(int argc, char* argv[])
{
    rad::Application app;
    if (!app.Init(argc, argv))
    {
        std::cerr << "Init failed!" << std::endl;
        return -1;
    }

    argparse::ArgumentParser program("MLTests");
    program.add_argument("--backend").default_value("CPU").store_into(g_options.backend);
    program.add_argument("--device-index").default_value(0).store_into(g_options.deviceIndex);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    ML::Initialize();
    rad::Ref<ML::Backend> backend;
    if (rad::StrCaseEqual(g_options.backend, "Vulkan"))
    {
        if (backend = ML::CreateVulkanBackend())
        {
            ML::RegisterBackend("Vulkan", backend);
            ML_LOG(info, "Vulkan backend initialized.");
        }
        else
        {
            ML_LOG(err, "Failed to create the Vulkan backend!");
            return -1;
        }
    }
    else if (!ML::GetBackend(g_options.backend))
    {
        ML_LOG(err, "Backend '{}' is not supported!", g_options.backend);
        return -1;
    }

    if (backend)
    {
        if (ML::Device* device = backend->GetDevice(g_options.deviceIndex))
        {
            ML::SetCurrentDevice(device);
            ML_LOG(info, "Set current device {}#{}: {}", backend->GetName(),
                g_options.deviceIndex, device->GetName());
        }
        else
        {
            ML_LOG(err, "{}: Device#{} is not available!", backend->GetName(), g_options.deviceIndex);
            return -1;
        }
    }
    else
    {
        return -1;
    }

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
