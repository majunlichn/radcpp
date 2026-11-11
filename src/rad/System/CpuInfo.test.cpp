#include <rad/System/CpuInfo.h>

#include <gtest/gtest.h>

#include <iostream>

TEST(CpuInfo, BrandString)
{
    const std::string_view brand = rad::GetCpuBrandString();
    std::cout << "CPU: " << brand << '\n';
    EXPECT_FALSE(brand.empty());
}
