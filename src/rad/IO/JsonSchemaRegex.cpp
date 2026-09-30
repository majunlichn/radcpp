#include "JsonSchemaRegex.h"

#include <rad/Core/Unicode.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

#ifndef RAD_JSON_SCHEMA_USE_STD_REGEX
#define RAD_JSON_SCHEMA_USE_STD_REGEX 0
#endif

#if RAD_JSON_SCHEMA_USE_STD_REGEX
#include <regex>
#else
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#endif

namespace rad::detail
{
namespace
{

#if !RAD_JSON_SCHEMA_USE_STD_REGEX
[[nodiscard]] std::string RegexErrorMessage(int code)
{
    std::array<PCRE2_UCHAR, 256> buffer{};
    const int length = pcre2_get_error_message(code, buffer.data(), buffer.size());
    if (length < 0)
    {
        return "PCRE2 error " + std::to_string(code);
    }
    return {reinterpret_cast<const char*>(buffer.data()), static_cast<std::size_t>(length)};
}

[[nodiscard]] std::string RegexCharacterRange(char32_t first, char32_t last)
{
    const auto character = [](char32_t value) {
        if (value <= 0x7f)
        {
            constexpr char digits[] = "0123456789abcdef";
            std::string escaped = "\\x";
            escaped += digits[(value >> 4) & 0xf];
            escaped += digits[value & 0xf];
            return escaped;
        }
        return Utf32ToUtf8(std::u32string_view(&value, 1));
    };
    auto range = character(first);
    if (first != last)
    {
        range += '-';
        range += character(last);
    }
    return range;
}

[[nodiscard]] const std::string& EcmaWhitespaceRanges(bool complement)
{
    constexpr std::pair<char32_t, char32_t> whitespace[] = {
        {0x0009, 0x000d}, {0x0020, 0x0020}, {0x00a0, 0x00a0},
        {0x1680, 0x1680}, {0x2000, 0x200a}, {0x2028, 0x2029},
        {0x202f, 0x202f}, {0x205f, 0x205f}, {0x3000, 0x3000},
        {0xfeff, 0xfeff},
    };
    static const auto ranges = [&] {
        std::array<std::string, 2> result;
        char32_t next = 0;
        for (const auto& [first, last] : whitespace)
        {
            result[0] += RegexCharacterRange(first, last);
            result[1] += RegexCharacterRange(next, first - 1);
            next = last + 1;
        }
        result[1] += RegexCharacterRange(next, 0x10ffff);
        return result;
    }();
    return ranges[complement ? 1 : 0];
}

[[nodiscard]] std::optional<std::string> NormalizeRegexPattern(std::string_view pattern)
{
    constexpr std::pair<std::string_view, std::string_view> categories[] = {
        {"Other", "C"},
        {"Control", "Cc"},
        {"Format", "Cf"},
        {"Unassigned", "Cn"},
        {"Private_Use", "Co"},
        {"Surrogate", "Cs"},
        {"Letter", "L"},
        {"Cased_Letter", "LC"},
        {"Lowercase_Letter", "Ll"},
        {"Modifier_Letter", "Lm"},
        {"Other_Letter", "Lo"},
        {"Titlecase_Letter", "Lt"},
        {"Uppercase_Letter", "Lu"},
        {"Mark", "M"},
        {"Spacing_Mark", "Mc"},
        {"Enclosing_Mark", "Me"},
        {"Nonspacing_Mark", "Mn"},
        {"Number", "N"},
        {"Decimal_Number", "Nd"},
        {"digit", "Nd"},
        {"Letter_Number", "Nl"},
        {"Other_Number", "No"},
        {"Punctuation", "P"},
        {"Connector_Punctuation", "Pc"},
        {"Dash_Punctuation", "Pd"},
        {"Close_Punctuation", "Pe"},
        {"Final_Punctuation", "Pf"},
        {"Initial_Punctuation", "Pi"},
        {"Other_Punctuation", "Po"},
        {"Open_Punctuation", "Ps"},
        {"Symbol", "S"},
        {"Currency_Symbol", "Sc"},
        {"Modifier_Symbol", "Sk"},
        {"Math_Symbol", "Sm"},
        {"Other_Symbol", "So"},
        {"Separator", "Z"},
        {"Line_Separator", "Zl"},
        {"Paragraph_Separator", "Zp"},
        {"Space_Separator", "Zs"},
    };

    std::string normalized;
    normalized.reserve(pattern.size());
    bool quoted = false;
    bool inClass = false;
    std::size_t classStart = 0;
    bool rangeHyphen = false;
    for (std::size_t index = 0; index < pattern.size(); ++index)
    {
        const char character = pattern[index];
        normalized += character;
        if (quoted)
        {
            if (character == '\\' && index + 1 < pattern.size() && pattern[index + 1] == 'E')
            {
                normalized += pattern[++index];
                quoted = false;
            }
            continue;
        }
        if (!inClass && character == '(' && pattern.substr(index, 3) == "(?#")
        {
            const auto end = pattern.find(')', index + 3);
            if (end != std::string_view::npos)
            {
                normalized.append(pattern.substr(index + 1, end - index));
                index = end;
                continue;
            }
        }
        if (character != '\\' || index + 1 == pattern.size())
        {
            if (inClass && character == '[' && index + 1 < pattern.size() &&
                (pattern[index + 1] == ':' || pattern[index + 1] == '.' ||
                 pattern[index + 1] == '='))
            {
                const char delimiter = pattern[index + 1];
                const auto end = pattern.find(std::string{delimiter, ']'}, index + 2);
                if (end != std::string_view::npos)
                {
                    normalized.append(pattern.substr(index + 1, end + 1 - index));
                    index = end + 1;
                    rangeHyphen = false;
                    continue;
                }
            }
            if (character == '[' && !inClass)
            {
                inClass = true;
                classStart = index;
            }
            else if (character == ']' && inClass)
            {
                inClass = false;
            }
            rangeHyphen = inClass && character == '-' && index > classStart + 1 &&
                          !(index == classStart + 2 && pattern[classStart + 1] == '^');
            continue;
        }
        const char escape = pattern[++index];
        normalized += escape;
        if (escape == 's' || escape == 'S')
        {
            if (inClass &&
                (rangeHyphen ||
                 (index + 2 < pattern.size() && pattern[index + 1] == '-' &&
                  pattern[index + 2] != ']')))
            {
                return std::nullopt;
            }
            normalized.resize(normalized.size() - 2);
            if (!inClass)
            {
                normalized += '[';
            }
            normalized += EcmaWhitespaceRanges(escape == 'S');
            if (!inClass)
            {
                normalized += ']';
            }
        }
        else if (escape == 'c' && index + 1 < pattern.size())
        {
            normalized += pattern[++index];
        }
        else if (escape == 'Q')
        {
            quoted = true;
        }
        else if ((escape == 'p' || escape == 'P') &&
                 index + 1 < pattern.size() && pattern[index + 1] == '{')
        {
            const auto end = pattern.find('}', index + 2);
            if (end == std::string_view::npos)
            {
                continue;
            }
            auto property = pattern.substr(index + 2, end - index - 2);
            if (property.starts_with("General_Category="))
            {
                property.remove_prefix(std::string_view("General_Category=").size());
            }
            else if (property.starts_with("gc="))
            {
                property.remove_prefix(3);
            }
            const auto category = std::find_if(
                std::begin(categories), std::end(categories),
                [property](const auto& candidate) { return candidate.first == property; });
            normalized += '{';
            normalized += category == std::end(categories) ? property : category->second;
            normalized += '}';
            index = end;
        }
        rangeHyphen = false;
    }
    return normalized;
}
#endif

} // namespace

struct JsonSchemaRegex::Impl
{
#if RAD_JSON_SCHEMA_USE_STD_REGEX
    using Expression = std::regex;
#else
    using Expression = std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)>;
#endif

    explicit Impl(Expression expression) : expression(std::move(expression))
    {
    }

    Expression expression;
};

JsonSchemaRegex::JsonSchemaRegex(std::unique_ptr<Impl> impl) : m_impl(std::move(impl))
{
}
JsonSchemaRegex::JsonSchemaRegex(JsonSchemaRegex&&) noexcept = default;
JsonSchemaRegex& JsonSchemaRegex::operator=(JsonSchemaRegex&&) noexcept = default;
JsonSchemaRegex::~JsonSchemaRegex() = default;

Result<JsonSchemaRegex, JsonSchemaRegexError>
JsonSchemaRegex::Compile(std::string_view pattern)
{
#if RAD_JSON_SCHEMA_USE_STD_REGEX
    if (pattern.find("\\p{") != std::string_view::npos ||
        pattern.find("\\P{") != std::string_view::npos)
    {
        return Failure(JsonSchemaRegexError{
            true, false, "Unicode property escapes are not supported by std::regex"});
    }
    try
    {
        return Success(JsonSchemaRegex(
            std::make_unique<Impl>(std::regex(std::string(pattern)))));
    }
    catch (const std::regex_error&)
    {
        return Failure(JsonSchemaRegexError{
            false, false, "pattern is not a valid regular expression"});
    }
#else
    int errorCode = 0;
    PCRE2_SIZE errorOffset = 0;
    // Preserve ASCII \d and \w by leaving PCRE2_UCP disabled.
    constexpr std::uint32_t options =
        PCRE2_UTF | PCRE2_ALT_BSUX | PCRE2_MATCH_UNSET_BACKREF | PCRE2_DOLLAR_ENDONLY;
    const auto normalized = NormalizeRegexPattern(pattern);
    if (!normalized)
    {
        return Failure(JsonSchemaRegexError{
            false, false, "whitespace escape cannot be a character range endpoint"});
    }
    Impl::Expression expression(
        pcre2_compile(reinterpret_cast<PCRE2_SPTR>(normalized->data()),
                      normalized->size(), options, &errorCode, &errorOffset, nullptr),
        &pcre2_code_free);
    if (!expression)
    {
        return Failure(JsonSchemaRegexError{
            false, false,
            "pattern is not a valid regular expression at normalized byte " +
                std::to_string(errorOffset) + ": " + RegexErrorMessage(errorCode)});
    }
    return Success(JsonSchemaRegex(std::make_unique<Impl>(std::move(expression))));
#endif
}

Result<bool, JsonSchemaRegexError> JsonSchemaRegex::Matches(
    std::string_view value, const JsonSchemaRegexMatchLimits& limits) const
{
#if RAD_JSON_SCHEMA_USE_STD_REGEX
    const JsonSchemaRegexMatchLimits defaults;
    if (limits.matchLimit != defaults.matchLimit ||
        limits.backtrackingDepthLimit != defaults.backtrackingDepthLimit ||
        limits.heapLimitKiB != defaults.heapLimitKiB)
    {
        return Failure(JsonSchemaRegexError{
            true, true,
            "custom regular expression matching limits are not supported by std::regex"});
    }
    try
    {
        return Success(std::regex_search(value.begin(), value.end(), m_impl->expression));
    }
    catch (const std::regex_error& error)
    {
        return Failure(JsonSchemaRegexError{
            false, true,
            std::string("regular expression matching failed: ") + error.what()});
    }
#else
    const auto& expression = m_impl->expression;
    const std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> matchData(
        pcre2_match_data_create_from_pattern(expression.get(), nullptr),
        &pcre2_match_data_free);
    const std::unique_ptr<pcre2_match_context, decltype(&pcre2_match_context_free)> context(
        pcre2_match_context_create(nullptr), &pcre2_match_context_free);
    if (!matchData || !context)
    {
        return Failure(JsonSchemaRegexError{
            false, true, "unable to allocate regular expression matching state"});
    }
    if (pcre2_set_match_limit(context.get(), limits.matchLimit) != 0 ||
        pcre2_set_depth_limit(context.get(), limits.backtrackingDepthLimit) != 0 ||
        pcre2_set_heap_limit(context.get(), limits.heapLimitKiB) != 0)
    {
        return Failure(JsonSchemaRegexError{
            false, true, "unable to set regular expression matching limits"});
    }
    const int result = pcre2_match(
        expression.get(),
        reinterpret_cast<PCRE2_SPTR>(value.empty() ? "" : value.data()), value.size(),
        0, 0, matchData.get(), context.get());
    if (result >= 0)
    {
        return Success(true);
    }
    if (result != PCRE2_ERROR_NOMATCH)
    {
        return Failure(JsonSchemaRegexError{
            false, true, "regular expression matching failed: " + RegexErrorMessage(result)});
    }
    return Success(false);
#endif
}

} // namespace rad::detail
