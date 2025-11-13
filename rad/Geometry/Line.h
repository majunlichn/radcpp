#pragma once

#include <rad/Geometry/Point.h>

namespace rad
{

template <std::floating_point T>
struct Line2D
{
    using value_type = T;
    Point2D<T> p1;
    Point2D<T> p2;
}; // struct Line2D

template <std::floating_point T>
struct Line3D
{
    using value_type = T;
    Point3D<T> p1;
    Point3D<T> p2;
}; // struct Line2D

} // namespace rad
