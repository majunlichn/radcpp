#include <rad/Core/Float.h>

#include <gtest/gtest.h>

TEST(Core, Float)
{
    float a = 1.0f;
    float b = rad::float_advance(a, 1);
    float d = rad::float_distance(a, b);
    EXPECT_EQ(d, 1.0f);
}
