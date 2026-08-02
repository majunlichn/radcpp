#include <rad/Diagnostics/StackTrace.h>

#include <rad/Core/Platform.h>

#include <boost/stacktrace/stacktrace.hpp>

#include <limits>

namespace rad
{

std::string GetStackTrace(std::size_t maxDepth, std::size_t skipFrames)
{
    if ((maxDepth == 0) || (skipFrames == std::numeric_limits<std::size_t>::max()))
    {
        return {};
    }

    // Boost.Stacktrace may include its own constructor/init frames. Skip those plus this
    // function so the first reported frame is the caller of GetStackTrace.
#if defined(RAD_OS_WINDOWS)
    constexpr std::size_t skipImplementationFrames = 3;
#else
    constexpr std::size_t skipImplementationFrames = 1;
#endif
    return boost::stacktrace::to_string(
        boost::stacktrace::stacktrace(skipImplementationFrames + skipFrames, maxDepth));
}

} // namespace rad
