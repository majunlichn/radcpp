#pragma once

#include <rad/Common/Float.h>
#include <rad/Common/Math.h>
#include <rad/Common/Numeric.h>
#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>

namespace rad
{

template <glm::length_t NumDim, typename T>
using Vec = glm::vec<NumDim, T, glm::defaultp>;

template <std::floating_point T>
using Vec2D = Vec<2, T>;
template <std::floating_point T>
using Vec3D = Vec<3, T>;

template <glm::length_t NumDim, typename T>
using Point = glm::vec<NumDim, T, glm::defaultp>;

template <std::floating_point T>
using Point2D = Vec<2, T>;
template <std::floating_point T>
using Point3D = Vec<3, T>;

template <glm::length_t NumDim, std::floating_point T>
bool IsAlmostZero(const Vec<NumDim, T>& v, T epsilon = std::numeric_limits<T>::epsilon() / 2)
{
    for (glm::length_t i = 0; i < NumDim; ++i)
    {
        if (glm::abs(v[i]) > epsilon)
        {
            return false;
        }
    }
    return true;
}

template <glm::length_t NumDim, std::floating_point T>
bool IsAlmostOrigin(const Point<NumDim, T>& v, T epsilon = std::numeric_limits<T>::epsilon() / 2)
{
    for (glm::length_t i = 0; i < NumDim; ++i)
    {
        if (glm::abs(v[i]) > epsilon)
        {
            return false;
        }
    }
    return true;
}

// @param epsilon: the min sine of the angle between the vectors (b - a) and (c - a).
template <glm::length_t NumDim, std::floating_point T>
bool IsCollinear(const Point<NumDim, T>& a, const Point<NumDim, T>& b, const Point<NumDim, T>& c, T epsilon = std::numeric_limits<T>::epsilon())
{
    Vec<NumDim, T> u = b - a;
    Vec<NumDim, T> v = c - a;
    T uNorm2 = glm::dot(u, u);
    T vNorm2 = glm::dot(v, v);
    constexpr T almostZero = std::numeric_limits<T>::epsilon() / 2;
    if ((uNorm2 <= almostZero) || (vNorm2 <= almostZero))
    {
        return true;
    }
    Vec<NumDim, T> x = glm::cross(u, v);
    T xNorm2 = glm::dot(x, x);
    return xNorm2 <= (uNorm2 * vNorm2 * epsilon * epsilon);
}

} // namespace rad
