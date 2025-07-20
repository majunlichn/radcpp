#include <rad/IO/Format.h>

namespace rad
{

std::string ToString(rad::ArrayRef<size_t> numbers)
{
    std::string str = "[ ";
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        str += std::to_string(numbers[i]);
        if (i < numbers.size() - 1)
        {
            str += ", ";
        }
    }
    str += " ]";
    return str;
}

} // namespace rad
