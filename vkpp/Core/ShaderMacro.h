#pragma once

#include <rad/Core/String.h>

namespace vkpp
{

struct ShaderMacro
{
    ShaderMacro()
    {
    }

    ShaderMacro(std::string_view name)
    {
        this->m_name = name;
    }

    ShaderMacro(std::string_view name, std::string_view value)
    {
        this->m_name = name;
        this->m_value = value;
    }

    std::string m_name;
    std::string m_value;

}; // class ShaderMacro

} // namespace vkpp
