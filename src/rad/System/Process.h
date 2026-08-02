#pragma once

#include <string>
#include <vector>

namespace rad
{

class Process
{
public:
    [[nodiscard]] static std::string ExecuteAndCaptureOutput(const std::string& executable,
                                                             const std::vector<std::string>& args);
}; // class Process

} // namespace rad
