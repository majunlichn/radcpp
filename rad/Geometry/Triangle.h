#pragma once

#include <rad/Geometry/Point.h>
#include <rad/Geometry/Line.h>

namespace rad
{

template <std::floating_point T>
struct Triangle2D
{
    using value_type = T;
    Point2D<T> p1;
    Point2D<T> p2;
    Point2D<T> p3;
}; // struct Triangle2D

template <std::floating_point T>
struct Triangle3D
{
    using value_type = T;
    Point3D<T> p1;
    Point3D<T> p2;
    Point3D<T> p3;
}; // struct Triangle3D

template<std::floating_point T>
T GetPerimeter(const Triangle2D<T>& tri)
{
    return glm::distance(tri.p1, tri.p2) + glm::distance(tri.p1, tri.p3) + glm::distance(tri.p2, tri.p3);
}

template<std::floating_point T>
T GetPerimeter(const Triangle3D<T>& tri)
{
    return glm::distance(tri.p1, tri.p2) + glm::distance(tri.p1, tri.p3) + glm::distance(tri.p2, tri.p3);
}

// Calculate triangle area with Heron's formula.
template<std::floating_point T>
T GetArea(const Triangle2D<T>& tri)
{
    T a = glm::distance(tri.p1, tri.p2);
    T b = glm::distance(tri.p1, tri.p3);
    T c = glm::distance(tri.p2, tri.p3);
    T p = (a + b + c) / 2;
    return glm::sqrt(p * (p - a) * (p - b) * (p - c));
}

// Calculate triangle area with Heron's formula.
template<std::floating_point T>
T GetArea(const Triangle3D<T>& tri)
{
    T a = glm::distance(tri.p1, tri.p2);
    T b = glm::distance(tri.p1, tri.p3);
    T c = glm::distance(tri.p2, tri.p3);
    T p = (a + b + c) / 2;
    return glm::sqrt(p * (p - a) * (p - b) * (p - c));
}

} // namespace rad
