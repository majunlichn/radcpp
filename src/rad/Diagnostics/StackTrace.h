#pragma once

#include <cstddef>
#include <string>

namespace rad
{

// Captures and formats up to maxDepth frames from the current thread. Implementation frames for
// GetStackTrace itself are always omitted. skipFrames omits additional frames above the caller
// (for example a wrapper or Exception constructor). Returns an empty string when maxDepth is zero.
[[nodiscard]] std::string GetStackTrace(std::size_t maxDepth = 32, std::size_t skipFrames = 0);

} // namespace rad
