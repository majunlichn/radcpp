#pragma once

#include <rad/Common/Float.h>
#include <rad/Common/Math.h>
#include <glm/glm.hpp>

namespace rad
{

template <std::floating_point T>
using Point2D = glm::vec<2, T, glm::defaultp>;
template <std::floating_point T>
using Point3D = glm::vec<3, T, glm::defaultp>;
template <std::floating_point T>
using Point4D = glm::vec<4, T, glm::defaultp>;

} // namespace rad
