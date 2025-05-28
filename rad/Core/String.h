#pragma once

#include <rad/Core/Platform.h>
#include <rad/Core/pystring.h>

#include <algorithm>
#include <format>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include <regex>
#include <sstream>

#include <boost/locale.hpp>
#include <boost/algorithm/string.hpp>

#if defined(_WIN32)
#define strcasecmp(a, b) _stricmp(a, b)
#define strncasecmp(a, b) _strnicmp(a, b)
#endif

namespace rad
{

// Treat std::string as UTF-8 encoded.

// Character Set Conversions:
using boost::locale::conv::to_utf;
using boost::locale::conv::from_utf;
using boost::locale::conv::utf_to_utf;

std::string StrPrintf(std::string_view format, ...);
std::string StrPrintfV(std::string_view format, va_list args);

bool StrEqual(std::string_view str1, std::string_view str2);
bool StrCaseEqual(std::string_view str1, std::string_view str2);

std::string StrUpper(std::string_view s);
std::string StrLower(std::string_view s);
void StrUpperInPlace(std::string& s);
void StrLowerInPlace(std::string& s);

std::string StrTrim(std::string_view str, std::string_view charlist = " \t\n\v\f\r");
void StrTrimInPlace(std::string& str, std::string_view charlist = " \t\n\v\f\r");

std::string StrRemovePrefix(std::string_view str, std::string_view prefix);
std::string StrRemoveSuffix(std::string_view str, std::string_view suffix);
void StrRemovePrefixInPlace(std::string& str, std::string_view prefix);
void StrRemoveSuffixInPlace(std::string& str, std::string_view suffix);

std::string StrFromWide(std::wstring_view wstr);
std::wstring StrToWide(std::string_view str);

bool IsDigit(char c);
bool IsHexDigit(char c);
bool StrIsDecInteger(std::string_view str);
bool StrIsHex(std::string_view str);
bool StrIsBin(std::string_view str);
// Check whether this string is a valid numeric string (a base 10 real number).
bool StrIsNumeric(std::string_view str);

bool StrToBool(std::string_view str);

std::vector<std::string> StrSplit(
    std::string_view str, std::string_view delimiters, bool skipEmptySubStr = true);
std::vector<std::string_view> StrSplitViews(
    std::string_view str, std::string_view delimiters, bool skipEmptySubStr = true);

bool StrReplaceFirst(std::string& str, std::string_view target, std::string_view rep);
bool StrReplaceLast(std::string& str, std::string_view target, std::string_view rep);
// Returns the replaced count.
// https://github.com/pytorch/pytorch/blob/main/c10/util/StringUtil.cpp
size_t StrReplaceAll(std::string& s, std::string_view from, std::string_view to);

static inline bool Contains(std::string_view str, std::string_view sub)
{
    return (str.find(sub) != std::string::npos);
}

static inline bool Contains(std::string_view str, const char ch)
{
    return (str.find(ch) != std::string::npos);
}

static inline void StrReverse(std::string& str)
{
    std::reverse(str.begin(), str.end());
}

struct StringLess
{
    using is_transparent = void;
    bool operator()(std::string_view left, std::string_view right) const
    {
        return (strcmp(left.data(), right.data()) < 0);
    }
};

// Case-insensitive compare op for std::set/map.
struct StringLessCaseInsensitive
{
    using is_transparent = void;
    bool operator()(std::string_view left, std::string_view right) const
    {
        return (strcasecmp(left.data(), right.data()) < 0);
    }
};

bool RegexMatch(const std::string& str, const std::regex& expr);
std::vector<std::string> RegexSplit(const std::string& str, const std::regex& expr);

template<std::ranges::range Container>
inline std::string JoinToString(const Container& tokens, const std::string& delim)
{
    std::ostringstream ss;
    for (auto it = tokens.begin(); it != tokens.end(); it++)
    {
        if (it != tokens.begin())
        {
            ss << delim;
        }
        ss << *it;
    }
    return ss.str();
}

} // namespace rad
