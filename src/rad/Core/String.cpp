#include <rad/Core/String.h>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <optional>

namespace rad
{

std::vector<std::string> StrSplit(std::string_view value, std::string_view delimiters,
                                  bool skipEmpty)
{
    const std::string input{value};
    const std::string delimiterChars{delimiters};
    const auto tokenCompression =
        skipEmpty ? boost::algorithm::token_compress_on : boost::algorithm::token_compress_off;

    std::vector<std::string> parts;
    boost::algorithm::split(parts, input, boost::algorithm::is_any_of(delimiterChars),
                            tokenCompression);
    if (skipEmpty)
    {
        std::erase_if(parts, [](const std::string& part) { return part.empty(); });
    }

    return parts;
}

std::string StrReplaceAll(std::string_view value, std::string_view search,
                          std::string_view replacement)
{
    std::string result{value};
    if (search.empty())
    {
        return result;
    }

    boost::algorithm::replace_all(result, std::string{search}, std::string{replacement});
    return result;
}

bool StrEqual(std::string_view lhs, std::string_view rhs) noexcept
{
    return lhs == rhs;
}

bool StrCaseEqual(std::string_view lhs, std::string_view rhs)
{
    return boost::algorithm::iequals(lhs, rhs);
}

int StrCmp(std::string_view lhs, std::string_view rhs) noexcept
{
    const int result = lhs.compare(rhs);
    if (result == 0)
    {
        return 0;
    }

    return result < 0 ? -1 : 1;
}

int StrCaseCmp(std::string_view lhs, std::string_view rhs) noexcept
{
    const std::size_t commonSize = std::min(lhs.size(), rhs.size());

    for (std::size_t i = 0; i < commonSize; ++i)
    {
        const int lhsChar = std::tolower(static_cast<unsigned char>(lhs[i]));
        const int rhsChar = std::tolower(static_cast<unsigned char>(rhs[i]));
        if (lhsChar != rhsChar)
        {
            return lhsChar < rhsChar ? -1 : 1;
        }
    }

    if (lhs.size() == rhs.size())
    {
        return 0;
    }

    return lhs.size() < rhs.size() ? -1 : 1;
}

std::string StrUpper(std::string_view value)
{
    std::string result{value};
    boost::algorithm::to_upper(result);
    return result;
}

std::string StrLower(std::string_view value)
{
    std::string result{value};
    boost::algorithm::to_lower(result);
    return result;
}

std::string StrTrim(std::string_view value)
{
    std::string result{value};
    boost::algorithm::trim(result);
    return result;
}

std::optional<bool> StrToBool(std::string_view value)
{
    const std::string normalized = StrLower(StrTrim(value));
    if (normalized == "1" || normalized == "true" || normalized == "on")
    {
        return true;
    }
    if (normalized == "0" || normalized == "false" || normalized == "off")
    {
        return false;
    }
    return std::nullopt;
}

} // namespace rad
