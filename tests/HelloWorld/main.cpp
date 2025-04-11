#include <rad/Config.h>
#include <format>
#include <iostream>

#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
    std::cout
        << std::format("radcpp Version: {}.{}.{}",
            RAD_VERSION_MAJOR, RAD_VERSION_MINOR, RAD_VERSION_PATCH)
        << std::endl;

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
