#pragma once

#include <radcpp/Core/Platform.h>
#include <radcpp/Core/String.h>
#include <boost/process.hpp>

namespace rad
{

std::vector<std::string> ExecuteAndReadLines(const std::string& command);

} // namespace rad
