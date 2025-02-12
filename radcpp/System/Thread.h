#pragma once

#include <radcpp/Core/Platform.h>
#include <string>

namespace rad
{

bool SetThreadName(std::string name);
std::string GetThreadName();

} // namespace rad
