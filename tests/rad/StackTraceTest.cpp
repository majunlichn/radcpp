#include <rad/System/Application.h>
#include <sstream>

#include <gtest/gtest.h>

TEST(Core, StackTrace)
{
    rad::Application* app = rad::Application::GetInstance();
    std::stringstream stream;
    app->PrintStackTrace(stream, 2);
    RAD_LOG(info, "Test Core.StackTrace:  \n{}\n", stream.str());
}
