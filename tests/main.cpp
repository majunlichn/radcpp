#include <rad/System/Application.h>
#include <rad/System/OS.h>
#include <rad/IO/Logging.h>
#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
    rad::Application app;
    if (!app.Init(argc, argv))
    {
        RAD_LOG_DEFAULT(err, "rad::Application::Init failed!");
    }

    RAD_LOG_DEFAULT(info, "User Name: {}", rad::getlogin());
    for (auto& path: rad::get_exec_path())
    {
        RAD_LOG_DEFAULT(info, "Path: {}", path);
    }

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
