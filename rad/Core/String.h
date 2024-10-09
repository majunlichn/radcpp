#pragma once

#include <rad/Core/Platform.h>
#include <rad/Core/pystring.h>
#include <string_view>

#include <boost/locale.hpp>
#include <boost/algorithm/string.hpp>

#if defined(RAD_COMPILER_MSVC)
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#endif

namespace rad
{

// Use std::string as UTF-8 encoded.

// Character Set Conversions:
using boost::locale::conv::to_utf;
using boost::locale::conv::from_utf;
using boost::locale::conv::utf_to_utf;

bool StrEqual(std::string_view str1, std::string_view str2);
bool StrCaseEqual(std::string_view str1, std::string_view str2);

std::string StrUpper(std::string_view s);
std::string StrLower(std::string_view s);
void StrUpperInplace(std::string& s);
void StrLowerInplace(std::string& s);

std::string StrTrim(std::string_view str, std::string_view charlist = " \t\n\v\f\r");
void StrTrimInPlace(std::string& str, std::string_view charlist = " \t\n\v\f\r");

std::string StrFromWide(std::wstring_view wstr);
std::wstring StrToWide(std::string_view str);

bool IsDigit(char c);
bool IsHexDigit(char c);
bool StrIsDecInteger(std::string_view str);
bool StrIsHex(std::string_view str);
bool StrIsBin(std::string_view str);
// Check whether this string is a valid numeric string (a base 10 real number).
bool StrIsNumeric(std::string_view str);

std::vector<std::string> StrSplit(
    std::string_view str, std::string_view delimiters, bool skipEmptySubStr = true);

std::string StrReplace(std::string_view str, std::string_view subOld, std::string_view subNew);
void StrReplaceInPlace(std::string& str, std::string_view subOld, std::string_view subNew);

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

} // namespace rad
