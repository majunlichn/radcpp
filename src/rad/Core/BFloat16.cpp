#include <rad/Core/BFloat16.h>

#include <ostream>

namespace rad
{

std::ostream& operator<<(std::ostream& stream, BFloat16 value)
{
    return stream << static_cast<float>(value);
}

} // namespace rad
