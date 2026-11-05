#pragma once

#include <rad/Core/Span.h>

#include <algorithm>
#include <cctype>
#include <format>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace rad
{

enum class HexCase
{
    Lower,
    Upper,
};

[[nodiscard]] std::vector<std::string> StrSplit(std::string_view value, std::string_view delimiters,
                                                bool skipEmpty = false);
[[nodiscard]] std::string StrReplaceAll(std::string_view value, std::string_view search,
                                        std::string_view replacement);
[[nodiscard]] bool StrEqual(std::string_view lhs, std::string_view rhs) noexcept;
[[nodiscard]] bool StrCaseEqual(std::string_view lhs, std::string_view rhs);
[[nodiscard]] int StrCmp(std::string_view lhs, std::string_view rhs) noexcept;
[[nodiscard]] int StrCaseCmp(std::string_view lhs, std::string_view rhs) noexcept;
[[nodiscard]] std::string StrUpper(std::string_view value);
[[nodiscard]] std::string StrLower(std::string_view value);
[[nodiscard]] std::string StrTrim(std::string_view value);
[[nodiscard]] std::string ToHexString(Span<const std::byte> bytes,
                                      HexCase letterCase = HexCase::Lower);
// Accepts "1"/"true"/"on" and "0"/"false"/"off" (case-insensitive, trimmed).
[[nodiscard]] std::optional<bool> StrToBool(std::string_view value);

// Validates the ASCII preferred host-name syntax in RFC 1035 section 2.3.1; no DNS lookup.
// Each label starts with a letter, ends with a letter/digit, and contains only letters/digits/'-'.
// Both validators accept single-label names and one trailing root dot, but not the root alone.
// Limits: 63 octets per label, 253 total excluding the root dot (255 in DNS wire format).
// Punycode is accepted as ASCII syntax only; neither validator performs IDNA validation.
[[nodiscard]] bool IsDomainRFC1035(std::string_view value) noexcept;

// RFC 1123 section 2.1: also permits digit-leading labels; the final label must contain a letter.
[[nodiscard]] bool IsDomainRFC1123(std::string_view value) noexcept;

// Validates an ASCII RFC 5321 Mailbox, not a display name or SMTP path; no DNS lookup.
// Supports quoted local parts and IPv4/IPv6 literals (including single-group IPv6 compression).
// Limits: 64 wire octets for the local part, 63 per domain label, 254 for the whole mailbox.
[[nodiscard]] bool IsEmailRFC5321(std::string_view value) noexcept;

template <std::ranges::input_range R, typename Proj = std::identity>
[[nodiscard]] std::string RangeToString(R&& range, std::string_view sep = ", ", Proj proj = {})
{
    std::string str;
    if constexpr (std::ranges::sized_range<R>)
    {
        str.reserve(std::ranges::size(range) * (8 + sep.size()));
    }

    auto it = std::ranges::begin(range);
    const auto end = std::ranges::end(range);
    if (it == end)
    {
        return {};
    }

    auto strInserter = std::back_inserter(str);
    std::format_to(strInserter, "{}", std::invoke(proj, *it));
    ++it;
    for (; it != end; ++it)
    {
        str += sep;
        std::format_to(strInserter, "{}", std::invoke(proj, *it));
    }
    return str;
}

struct StringLess
{
    using is_transparent = void;

    bool operator()(std::string_view left, std::string_view right) const
    {
        return left.compare(right) < 0;
    }
};

struct StringLessCaseInsensitive
{
    using is_transparent = void;

    bool operator()(std::string_view left, std::string_view right) const
    {
        return std::ranges::lexicographical_compare(
            left, right,
            [](char a, char b)
            {
                return std::tolower(static_cast<unsigned char>(a)) <
                       std::tolower(static_cast<unsigned char>(b));
            });
    }
};

} // namespace rad
