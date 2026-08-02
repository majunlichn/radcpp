#include <rad/Core/Float8.h>

#include <ostream>

namespace rad
{

std::ostream& operator<<(std::ostream& stream, Float8E4M3 value)
{
    return stream << static_cast<float>(value);
}

std::ostream& operator<<(std::ostream& stream, Float8E5M2 value)
{
    return stream << static_cast<float>(value);
}

} // namespace rad
