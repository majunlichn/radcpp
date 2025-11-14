#pragma once

#include <rad/Geometry/Point.h>

namespace rad
{

template <std::floating_point T>
struct Sphere
{
    using value_type = T;
    Point3D<T> center;
    T radius;
};

template <std::floating_point T>
T GetVolume(const Sphere<T>& s)
{
    return (T(4) / T(3)) * glm::pi<T>() * s.radius * s.radius * s.radius;
}

template <std::floating_point T>
T GetSurfaceArea(const Sphere<T>& s)
{
    return T(4) * glm::pi<T>() * s.radius * s.radius;
}

// Check if point p is inside sphere s.
template <std::floating_point T>
bool IsInside(const Sphere<T>& s, const Point3D<T>& p)
{
    Vec3D<T> v = p - s.center;
    T dist2 = glm::dot(v, v);
    return dist2 <= (s.radius * s.radius);
}

// Check if sphere b is inside sphere a.
template <std::floating_point T>
bool IsInside(const Sphere<T>& a, const Sphere<T>& b)
{
    if (a.radius >= b.radius)
    {
        T d = glm::distance(a.center, b.center);
        return ((d + b.radius) <= a.radius);
    }
    else
    {
        return false;
    }
}

template <std::floating_point T>
bool HasIntersection(const Sphere<T>& a, const Sphere<T>& b)
{
    Vec3D<T> v = b.center - a.center;
    T dist2 = glm::dot(v, v);
    T r = a.radius + b.radius;
    return dist2 <= (r * r);
}

} // namespace rad
