#pragma once

#include <radcpp/Core/Platform.h>
#include <ctime>
#include <chrono>

namespace rad
{

struct tm* LocalTime(const time_t* timer, struct tm* buffer);
// Returns string in format "YYYY-MM-DDThh:mm:ssZ" or empty if failed.
std::string GetTimeStringUTC();
// Returns string in format "YYYY-MM-DDThh:mm:ss+0000" or empty if failed.
std::string GetTimeStringISO8601();

} // namespace rad
