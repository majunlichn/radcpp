#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace rad
{

[[nodiscard]] bool SetThreadName(std::string_view name);
[[nodiscard]] std::string GetThreadName();
[[nodiscard]] std::uint64_t GetCurrentThreadId();

} // namespace rad
