#pragma once

#include <rad/Geometry/Point.h>

namespace rad
{

template <std::floating_point T>
struct Circle
{
    using value_type = T;
    Point2D<T> center;
    T radius;
};

template <std::floating_point T>
T GetPerimeter(const Circle<T>& c)
{
    return T(2) * glm::pi<T>() * c.radius;
}

template <std::floating_point T>
T GetArea(const Circle<T>& c)
{
    return glm::pi<T>() * c.radius * c.radius;
}

} // namespace rad
