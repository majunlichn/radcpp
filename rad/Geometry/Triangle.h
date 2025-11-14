#pragma once

#include <rad/Geometry/Point.h>
#include <rad/Geometry/Line.h>

namespace rad
{

template <glm::length_t NumDim, std::floating_point T>
struct Triangle
{
    using value_type = T;
    Point<NumDim, T> p1;
    Point<NumDim, T> p2;
    Point<NumDim, T> p3;
}; // struct Triangle

template <std::floating_point T>
using Triangle2D = Vec<2, T>;
template <std::floating_point T>
using Triangle3D = Vec<3, T>;

template<glm::length_t NumDim, std::floating_point T>
bool IsDegenerate(const Triangle<NumDim, T>& tri)
{
    return IsCollinear(tri.p1, tri.p2, tri.p3);
}

template<glm::length_t NumDim, std::floating_point T>
T GetPerimeter(const Triangle<NumDim, T>& tri)
{
    return glm::distance(tri.p1, tri.p2) + glm::distance(tri.p1, tri.p3) + glm::distance(tri.p2, tri.p3);
}

// Calculate triangle area with Heron's formula.
template<glm::length_t NumDim, std::floating_point T>
T GetArea(const Triangle<NumDim, T>& tri)
{
    T a = glm::distance(tri.p1, tri.p2);
    T b = glm::distance(tri.p1, tri.p3);
    T c = glm::distance(tri.p2, tri.p3);
    T p = (a + b + c) / 2;
    return glm::sqrt(p * (p - a) * (p - b) * (p - c));
}

} // namespace rad
