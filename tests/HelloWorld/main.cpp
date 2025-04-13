#include <rad/System/Application.h>
#include <format>
#include <iostream>

#include <gtest/gtest.h>

void TestStackTrace()
{
    rad::Application* app = rad::Application::GetInstance();
    app->PrintStackTrace();
}

int main(int argc, char* argv[])
{
    rad::Application app;
    if (!app.Init(argc, argv))
    {
        std::cerr << "Init failed!" << std::endl;
        return -1;
    }

    testing::InitGoogleTest(&argc, argv);
    TestStackTrace();
    return RUN_ALL_TESTS();
}
