#pragma once

#include <rad/Geometry/Point.h>

namespace rad
{

template <glm::length_t NumDim, std::floating_point T>
struct Line
{
    using value_type = T;
    Point<NumDim, T> p1;
    Point<NumDim, T> p2;
};

} // namespace rad
