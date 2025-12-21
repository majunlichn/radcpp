#include <rad/Geometry/Point.h>
#include <rad/Geometry/Line.h>
#include <rad/Geometry/Triangle.h>

#include <gtest/gtest.h>

TEST(Geometry, Line)
{
    rad::Line3D<float> line;
    line.p1 = { 1, 2, 3 };
    line.p2 = { 4, 6, 9 };
    rad::Point3D<float> p = { 2, 5, 7 };
    float dist = rad::GetDistance(line, p);
    EXPECT_TRUE(rad::AlmostEqual(dist, std::sqrt(65.f / 61.f)));
}
