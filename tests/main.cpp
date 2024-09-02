#include <rad/System/Application.h>
#include <rad/IO/Logging.h>
#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
    rad::Application app;
    if (!app.Init(argc, argv))
    {
        RAD_LOG_DEFAULT(err, "rad::Application::Init failed!");
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
