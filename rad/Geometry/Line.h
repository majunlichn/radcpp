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

template <std::floating_point T>
using Line2D = Line<2, T>;
template <std::floating_point T>
using Line3D = Line<3, T>;

template <glm::length_t NumDim, std::floating_point T>
bool IsDegenerate(const Line<NumDim, T>& line, T epsilon = std::numeric_limits<T>::epsilon() / 2)
{
    Vec<NumDim> v = glm::abs(line.p2 - line.p1);
    for (glm::length_t i = 0; i < NumDim; ++i)
    {
        if (v[i] > epsilon)
        {
            return false;
        }
    }
    return true;
}

template <glm::length_t NumDim, std::floating_point T>
T GetLength(const Line<NumDim, T>& line)
{
    return glm::distance(line.p1, line.p2);
}

} // namespace rad
