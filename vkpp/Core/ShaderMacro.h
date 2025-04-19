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

    template<typename T>
    ShaderMacro(std::string_view name, T definition)
    {
        this->m_name = name;
        this->m_definition = std::to_string(definition);
    }

    std::string m_name;
    std::string m_definition;

}; // class ShaderMacro

} // namespace vkpp
