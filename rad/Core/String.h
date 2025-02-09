#pragma once

#include <rad/Core/Platform.h>
// Pystring is a collection of C++ functions which match the interface
// and behavior of python's string class methods using std::string.
#include <rad/Core/pystring.h>
#include <string_view>

#include <boost/locale.hpp>
#include <boost/algorithm/string.hpp>

// https://github.com/nemtrif/utfcpp
#include <utf8.h>

#include <regex>
#include <sstream>

#if defined(RAD_COMPILER_MSVC)
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#endif

namespace rad
{

// Treat std::string as UTF-8 encoded.

// Character Set Conversions:
using boost::locale::conv::to_utf;
using boost::locale::conv::from_utf;
using boost::locale::conv::utf_to_utf;

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

std::string ToString(std::wstring_view wstr);
std::wstring ToWideString(std::string_view str);

inline const char8_t* UTF8Cast(std::string_view str)
{
    return (const char8_t*)str.data();
}

bool IsDigit(char c);
bool IsHexDigit(char c);
bool StrIsDecInteger(std::string_view str);
bool StrIsHex(std::string_view str);
bool StrIsBin(std::string_view str);
// Check whether this string is a valid numeric string (a base 10 real number).
bool StrIsNumeric(std::string_view str);

std::vector<std::string> StrSplit(
    std::string_view str, std::string_view delimiters, bool skipEmptySubStr = true);
std::vector<std::string_view> ViewSplit(
    std::string_view str, std::string_view delimiters, bool skipEmptySubStr = true);

std::string StrReplace(std::string_view str, std::string_view subOld, std::string_view subNew, int count = -1);
void StrReplaceInPlace(std::string& str, std::string_view subOld, std::string_view subNew, int count = -1);
bool StrReplaceFirst(std::string& str, std::string_view target, std::string_view rep);
bool StrReplaceLast(std::string& str, std::string_view target, std::string_view rep);
// PyTorch version: returns the replaced count.
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

static inline void Reverse(std::string& str)
{
    std::reverse(str.begin(), str.end());
}

static inline std::string ReverseCopy(std::string str)
{
    std::reverse(str.begin(), str.end());
    return str;
}

static inline std::string Repeat(const std::string& str, size_t n)
{
    std::string result;
    for (size_t i = 0; i < n; ++i)
    {
        result += str;
    }
    return result;
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

template<typename Container>
inline std::string JoinIntoString(const Container& tokens, const std::string& delim)
{
    std::ostringstream result;
    for (auto it = tokens.begin(); it != tokens.end(); it++)
    {
        if (it != tokens.begin())
        {
            result << delim;
        }
        result << *it;
    }
    return result.str();
}

} // namespace rad
