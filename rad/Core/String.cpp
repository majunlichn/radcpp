#include <rad/Core/String.h>
#if defined(RAD_OS_WINDOWS)
#include <Windows.h>
#endif

namespace rad
{

static std::locale g_locale(".UTF-8");

bool StrEqual(std::string_view str1, std::string_view str2)
{
    return (str1 == str2);
}

bool StrCaseEqual(std::string_view str1, std::string_view str2)
{
#if defined(RAD_COMPILER_MSVC)
    return (_stricmp(str1.data(), str2.data()) == 0);
#else
    return (strcasecmp(str1.data(), str2.data()) == 0);
#endif
}

std::string StrUpper(std::string_view s)
{
    std::string buffer(s);
    StrUpperInplace(buffer);
    return buffer;
}

std::string StrLower(std::string_view s)
{
    std::string buffer(s);
    StrLowerInplace(buffer);
    return buffer;
}

void StrUpperInplace(std::string& s)
{
    boost::algorithm::to_upper(s, g_locale);
}

void StrLowerInplace(std::string& s)
{
    boost::algorithm::to_lower(s, g_locale);
}

std::string StrTrim(std::string_view str, std::string_view charlist)
{
    size_t beg = str.find_first_not_of(charlist);
    size_t end = str.find_last_not_of(charlist) + 1;
    return std::string(str.substr(beg, end - beg));
}

void StrTrimInPlace(std::string& str, std::string_view charlist)
{
    str.erase(str.find_last_not_of(charlist) + 1);
    str.erase(0, str.find_first_not_of(charlist));
}

std::string StrFromWide(std::wstring_view wstr)
{
#if defined(RAD_OS_WINDOWS)
    std::string str;
    int charCount = ::WideCharToMultiByte(CP_UTF8,
        0, wstr.data(), static_cast<int>(wstr.length()), NULL, 0, NULL, NULL);
    if (charCount > 0)
    {
        str.resize(charCount, 0);
        ::WideCharToMultiByte(CP_UTF8,
            0, wstr.data(), static_cast<int>(wstr.length()), &str[0], charCount, NULL, NULL);
    }
    return str;
#else
    return utf_to_utf<char>(wstr.data());
#endif
}

std::wstring StrToWide(std::string_view str)
{
#if defined(RAD_OS_WINDOWS)
    std::wstring wstr;
    int charCount = ::MultiByteToWideChar(CP_UTF8,
        0, str.data(), static_cast<int>(str.length()), NULL, 0);
    if (charCount > 0)
    {
        wstr.resize(charCount, 0);
        ::MultiByteToWideChar(CP_UTF8,
            0, str.data(), static_cast<int>(str.length()), &wstr[0], charCount);
    }
    return wstr;
#else
    return utf_to_utf<wchar_t>(str.data());
#endif
}

bool IsDigit(char c)
{
    return (c >= '0' && c <= '9');
}

bool IsHexDigit(char c)
{
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
}

bool StrIsDecInteger(std::string_view str)
{
    if (str.empty())
    {
        return false;
    }
    size_t i = 0;
    if (str.size() >= 2)
    {
        if (str[0] == '+' || str[0] == '-')
        {
            i++;
        }
    }
    for (; i < str.size(); ++i)
    {
        if (!IsDigit(str[i]))
        {
            return false;
        }
    }
    return true;
}

bool StrIsHex(std::string_view str)
{
    if (str.starts_with("0x") || str.starts_with("0X"))
    {
        for (size_t i = 2; i < str.size(); ++i)
        {
            if (!IsHexDigit(str[i]))
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool StrIsBin(std::string_view str)
{
    if (str.starts_with("0b") || str.starts_with("0B"))
    {
        for (size_t i = 2; i < str.size(); ++i)
        {
            if ((str[i] != '0') && (str[i] != '1'))
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool StrIsNumeric(std::string_view str)
{
    const char* p = str.data();
    if ((*p == '-') || (*p == '+'))
    {
        ++p;
    }

    bool hasDot = false;
    while (*p != '\0')
    {
        if (*p == '.')
        {
            if (hasDot)
            {
                return false;
            }
            hasDot = true;
        }
        else if (!IsDigit(*p))
        {
            return false;
        }

        ++p;
    }

    return true;
}

std::vector<std::string> StrSplit(
    std::string_view str, std::string_view delimiters, bool skipEmptySubStr)
{
    std::vector<std::string> substrs;

    std::string::size_type pos = 0;
    std::string::size_type offset = 0;

    while (offset < str.length() + 1)
    {
        pos = str.find_first_of(delimiters, offset);
        if (pos == std::string::npos)
        {
            pos = str.length();
        }
        if (pos != offset || !skipEmptySubStr)
        {
            substrs.push_back(std::string(str.data() + offset, pos - offset));
        }
        offset = pos + 1;
    }

    return substrs;
}

std::string StrReplace(std::string_view str, std::string_view subOld, std::string_view subNew)
{
    std::string newStr;
    newStr.reserve(str.size());
    std::string::size_type offset = 0u;
    std::string::size_type pos = 0u;
    while ((pos = str.find(subOld, offset)) != std::string::npos)
    {
        newStr.append(str, offset, pos - offset);
        newStr.append(subNew);
        offset = pos + subOld.size();
    }
    if (offset < str.size())
    {
        pos = str.size();
        newStr.append(str, offset, pos - offset);
    }
    return newStr;
}

void StrReplaceInPlace(std::string& str, std::string_view subOld, std::string_view subNew)
{
    std::string::size_type pos = 0u;
    while ((pos = str.find(subOld, pos)) != std::string::npos)
    {
        str.replace(pos, subOld.length(), subNew);
        pos += subNew.length();
    }
}

} // namespace rad
