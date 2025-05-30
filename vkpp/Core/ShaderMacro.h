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

    ShaderMacro(std::string_view name, std::string_view definition)
    {
        this->m_name = name;
        this->m_definition = definition;
    }

    std::string m_name;
    std::string m_definition;

}; // class ShaderMacro

} // namespace vkpp
