#include <rad/IO/Logging.h>
#include <format>
#include <iostream>

#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
    std::string logFileName = std::string(argv[0]) + ".log";
    rad::InitLogging(logFileName, true);
    RAD_LOG(info, "radcpp Version: {}.{}.{}",
        RAD_VERSION_MAJOR, RAD_VERSION_MINOR, RAD_VERSION_PATCH);

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
