#include <radcpp/System/Application.h>
#include <radcpp/System/Thread.h>
#include <radcpp/IO/Logging.h>
#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
    rad::SetThreadName("main");
    rad::Application app;
    if (!app.Init(argc, argv))
    {
        SPDLOG_ERROR("rad::Application::Init failed!");
        return -1;
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
