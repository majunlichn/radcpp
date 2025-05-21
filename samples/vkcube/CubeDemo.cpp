#include "CubeDemo.h"

CubeDemo::CubeDemo(rad::Ref<vkpp::Instance> instance) :
    VulkanWindow(std::move(instance))
{
}

CubeDemo::~CubeDemo()
{
}

bool CubeDemo::Init(int argc, char* argv)
{
    return true;
}
