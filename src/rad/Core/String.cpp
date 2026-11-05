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
namespace
{

bool IsAsciiLetter(char c) noexcept
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool IsAsciiDigit(char c) noexcept
{
    return c >= '0' && c <= '9';
}

bool IsAsciiAlnum(char c) noexcept
{
    return IsAsciiLetter(c) || IsAsciiDigit(c);
}

bool IsDomainLabel(std::string_view label, bool allowLeadingDigit) noexcept
{
    if (label.empty() || label.size() > 63 || !IsAsciiAlnum(label.back()) ||
        !(allowLeadingDigit ? IsAsciiAlnum(label.front()) : IsAsciiLetter(label.front())))
    {
        return false;
    }
    return std::ranges::all_of(label, [](char c) { return IsAsciiAlnum(c) || c == '-'; });
}

bool IsDomainName(std::string_view value, bool allowLeadingDigit) noexcept
{
    if (!value.empty() && value.back() == '.')
    {
        value.remove_suffix(1);
    }
    if (value.empty() || value.size() > 253)
    {
        return false;
    }

    while (true)
    {
        const auto dot = value.find('.');
        const auto label = value.substr(0, dot);
        if (!IsDomainLabel(label, allowLeadingDigit))
        {
            return false;
        }
        if (dot == std::string_view::npos)
        {
            return !allowLeadingDigit || std::ranges::any_of(label, IsAsciiLetter);
        }
        value.remove_prefix(dot + 1);
    }
}

} // namespace

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

std::string ToHexString(Span<const std::byte> bytes, HexCase letterCase)
{
    constexpr char lowerDigits[] = "0123456789abcdef";
    constexpr char upperDigits[] = "0123456789ABCDEF";
    const char* digits = (letterCase == HexCase::Lower) ? lowerDigits : upperDigits;
    std::string result(bytes.size() * 2, '\0');
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
        const auto value = std::to_integer<unsigned char>(bytes[i]);
        result[i * 2] = digits[value >> 4];
        result[i * 2 + 1] = digits[value & 0x0f];
    }
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

bool IsDomainRFC1035(std::string_view value) noexcept
{
    return IsDomainName(value, false);
}

bool IsDomainRFC1123(std::string_view value) noexcept
{
    return IsDomainName(value, true);
}

bool IsEmailRFC5321(std::string_view value) noexcept
{
    if (value.size() > 254)
    {
        return false;
    }

    const auto at = value.rfind('@');
    if (at == std::string_view::npos || at == 0 || at > 64 || at + 1 == value.size())
    {
        return false;
    }

    const auto local = value.substr(0, at);
    if (local.front() == '"')
    {
        if (local.size() < 2 || local.back() != '"')
        {
            return false;
        }
        for (std::size_t i = 1; i < local.size() - 1; ++i)
        {
            char c = local[i];
            if (c == '\\')
            {
                if (++i == local.size() - 1)
                {
                    return false;
                }
                c = local[i];
            }
            else if (c == '"')
            {
                return false;
            }
            if (c < ' ' || c > '~')
            {
                return false;
            }
        }
    }
    else
    {
        bool atomStart = true;
        for (char c : local)
        {
            if (c == '.')
            {
                if (atomStart)
                {
                    return false;
                }
                atomStart = true;
            }
            else
            {
                if (!IsAsciiAlnum(c) &&
                    std::string_view{"!#$%&'*+-/=?^_`{|}~"}.find(c) == std::string_view::npos)
                {
                    return false;
                }
                atomStart = false;
            }
        }
        if (atomStart)
        {
            return false;
        }
    }

    auto domain = value.substr(at + 1);
    if (domain.front() == '[' && domain.back() == ']')
    {
        domain.remove_prefix(1);
        domain.remove_suffix(1);
        const auto isIPv4 = [](std::string_view address)
        {
            for (int part = 0; part < 4; ++part)
            {
                const auto dot = address.find('.');
                const auto octet = address.substr(0, dot);
                if (octet.empty() || octet.size() > 3)
                {
                    return false;
                }
                unsigned number = 0;
                for (char c : octet)
                {
                    if (!IsAsciiDigit(c))
                    {
                        return false;
                    }
                    number = number * 10 + (c - '0');
                }
                if (number > 255 || (dot == std::string_view::npos) != (part == 3))
                {
                    return false;
                }
                if (dot != std::string_view::npos)
                {
                    address.remove_prefix(dot + 1);
                }
            }
            return true;
        };
        if (domain.size() < 5 || (domain[0] != 'I' && domain[0] != 'i') ||
            (domain[1] != 'P' && domain[1] != 'p') || (domain[2] != 'V' && domain[2] != 'v') ||
            domain[3] != '6' || domain[4] != ':')
        {
            return isIPv4(domain);
        }
        domain.remove_prefix(5);
        if (domain.empty())
        {
            return false;
        }

        bool compressed = domain.starts_with("::");
        if (compressed)
        {
            domain.remove_prefix(2);
        }
        unsigned groups = 0;
        while (!domain.empty())
        {
            const auto colon = domain.find(':');
            const auto group = domain.substr(0, colon);
            if (group.find('.') != std::string_view::npos)
            {
                if (colon != std::string_view::npos || !isIPv4(group))
                {
                    return false;
                }
                groups += 2;
                break;
            }
            if (group.empty() || group.size() > 4)
            {
                return false;
            }
            for (char c : group)
            {
                if (!IsAsciiDigit(c) && !(c >= 'A' && c <= 'F') && !(c >= 'a' && c <= 'f'))
                {
                    return false;
                }
            }
            ++groups;
            if (colon == std::string_view::npos)
            {
                break;
            }
            domain.remove_prefix(colon + 1);
            if (domain.starts_with(':'))
            {
                if (compressed)
                {
                    return false;
                }
                compressed = true;
                domain.remove_prefix(1);
            }
            else if (domain.empty())
            {
                return false;
            }
        }
        return compressed ? groups < 8 : groups == 8;
    }

    while (!domain.empty())
    {
        const auto dot = domain.find('.');
        const auto label = domain.substr(0, dot);
        if (!IsDomainLabel(label, true))
        {
            return false;
        }
        if (dot == std::string_view::npos)
        {
            return true;
        }
        domain.remove_prefix(dot + 1);
    }
    return false;
}

} // namespace rad
