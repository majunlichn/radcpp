#include <rad/System/Application.h>

#include <gtest/gtest.h>

TEST(System, Application)
{
    auto& app = rad::Application::Instance();
    EXPECT_FALSE(app.Arguments().empty());
    EXPECT_NO_THROW(app.InstallDefaultTerminateHandler());
    EXPECT_NO_THROW(app.InstallDefaultSignalHandlers());
}
