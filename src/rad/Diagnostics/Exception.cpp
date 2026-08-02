#include <rad/Diagnostics/Exception.h>

#include <boost/exception/diagnostic_information.hpp>
#include <boost/exception/info.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace rad
{

Exception::Exception(std::string message, std::source_location location,
                     std::size_t stackTraceDepth) :
    m_message(std::move(message)),
    m_location(location),
    m_stackTraceDepth(stackTraceDepth),
    m_stackTrace(GetStackTrace(stackTraceDepth, 1))
{
    SetLocation(location);
}

const char* Exception::what() const noexcept
{
    return m_message.c_str();
}

const std::string& Exception::Message() const noexcept
{
    return m_message;
}

const std::source_location& Exception::Location() const noexcept
{
    return m_location;
}

const std::string& Exception::StackTrace() const noexcept
{
    return m_stackTrace;
}

std::string Exception::DiagnosticInformation() const
{
    std::string diagnostic = boost::diagnostic_information(*this);

    if (!m_stackTrace.empty())
    {
        diagnostic += "\nStack trace:\n";
        diagnostic += m_stackTrace;
    }

    return diagnostic;
}

void Exception::SetLocation(std::source_location location)
{
    m_location = location;

    const auto maxLine = static_cast<std::uint_least32_t>(std::numeric_limits<int>::max());
    const int line = static_cast<int>(std::min(m_location.line(), maxLine));
    *this << boost::throw_file(m_location.file_name()) << boost::throw_line(line)
          << boost::throw_function(m_location.function_name());
}

} // namespace rad
