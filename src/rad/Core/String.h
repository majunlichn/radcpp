#pragma once

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rad
{

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
// Accepts "1"/"true"/"on" and "0"/"false"/"off" (case-insensitive, trimmed).
[[nodiscard]] std::optional<bool> StrToBool(std::string_view value);

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
