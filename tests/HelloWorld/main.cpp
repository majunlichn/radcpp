#include <rad/Config.h>
#include <format>
#include <iostream>

int main(int argc, char* argv[])
{
    std::cout <<
        std::format("radcpp Version: {}.{}.{}",
            RAD_VERSION_MAJOR, RAD_VERSION_MINOR, RAD_VERSION_PATCH) << std::endl;
    return 0;
}
