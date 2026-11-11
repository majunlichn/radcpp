#include "JsonSchemaRegex.h"

#include <rad/Core/Unicode.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

namespace rad::detail
{
namespace
{

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

[[nodiscard]] std::string_view NormalizeRegexCategory(std::string_view property) noexcept
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
    return category == std::end(categories) ? property : category->second;
}

[[nodiscard]] std::optional<std::string> NormalizeRegexPattern(std::string_view pattern)
{
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
            normalized += '{';
            normalized += NormalizeRegexCategory(pattern.substr(index + 2, end - index - 2));
            normalized += '}';
            index = end;
        }
        rangeHyphen = false;
    }
    return normalized;
}

} // namespace

void JsonSchemaRegex::MatchContext::ContextDeleter::operator()(
    pcre2_real_match_context_8* context) const noexcept
{
    pcre2_match_context_free(context);
}

void JsonSchemaRegex::MatchContext::DataDeleter::operator()(
    pcre2_real_match_data_8* data) const noexcept
{
    pcre2_match_data_free(data);
}

JsonSchemaRegex::MatchContext::MatchContext(const JsonSchemaRegexLimits& limits) : m_limits(limits)
{
}

JsonSchemaRegex::MatchContext::~MatchContext() = default;

void JsonSchemaRegex::CodeDeleter::operator()(pcre2_real_code_8* expression) const noexcept
{
    pcre2_code_free(expression);
}

JsonSchemaRegex::JsonSchemaRegex(Expression expression, std::uint32_t matchPairs) :
    m_expression(std::move(expression)),
    m_matchPairs(matchPairs)
{
}
JsonSchemaRegex::JsonSchemaRegex(JsonSchemaRegex&&) noexcept = default;
JsonSchemaRegex& JsonSchemaRegex::operator=(JsonSchemaRegex&&) noexcept = default;
JsonSchemaRegex::~JsonSchemaRegex() = default;

Result<JsonSchemaRegex, JsonSchemaRegexError>
JsonSchemaRegex::Compile(std::string_view pattern)
{
    int errorCode = 0;
    PCRE2_SIZE errorOffset = 0;
    // Preserve ASCII \d and \w by leaving PCRE2_UCP disabled.
    constexpr std::uint32_t options =
        PCRE2_UTF | PCRE2_ALT_BSUX | PCRE2_MATCH_UNSET_BACKREF | PCRE2_DOLLAR_ENDONLY;
    const auto normalized = NormalizeRegexPattern(pattern);
    if (!normalized)
    {
        return Failure(JsonSchemaRegexError{
            "whitespace escape cannot be a character range endpoint"});
    }
    Expression expression(
        pcre2_compile(reinterpret_cast<PCRE2_SPTR>(normalized->data()),
                      normalized->size(), options, &errorCode, &errorOffset, nullptr));
    if (!expression)
    {
        return Failure(JsonSchemaRegexError{
            "pattern is not a valid regular expression at normalized byte " +
                std::to_string(errorOffset) + ": " + RegexErrorMessage(errorCode)});
    }
    std::uint32_t captures = 0;
    const int info = pcre2_pattern_info(expression.get(), PCRE2_INFO_CAPTURECOUNT, &captures);
    if (info != 0)
    {
        return Failure(JsonSchemaRegexError{
            "unable to inspect regular expression: " + RegexErrorMessage(info)});
    }
    return Success(JsonSchemaRegex(std::move(expression), captures + 1));
}

Result<bool, JsonSchemaRegexError> JsonSchemaRegex::Matches(
    std::string_view value, MatchContext& context) const
{
    if (!context.m_context)
    {
        MatchContext::ContextHandle state(pcre2_match_context_create(nullptr));
        if (!state)
        {
            return Failure(JsonSchemaRegexError{
                "unable to allocate regular expression matching state"});
        }
        const auto& limits = context.m_limits;
        // PCRE2 limit setters always return zero.
        pcre2_set_match_limit(state.get(), limits.matchLimit);
        pcre2_set_depth_limit(state.get(), limits.backtrackingDepthLimit);
        pcre2_set_heap_limit(state.get(), limits.heapLimitKiB);
        context.m_context = std::move(state);
    }
    if (context.m_capacity < m_matchPairs)
    {
        MatchContext::DataHandle data(pcre2_match_data_create(m_matchPairs, nullptr));
        if (!data)
        {
            return Failure(JsonSchemaRegexError{
                "unable to allocate regular expression matching state"});
        }
        context.m_data = std::move(data);
        context.m_capacity = m_matchPairs;
    }
    const int result = pcre2_match(
        m_expression.get(),
        reinterpret_cast<PCRE2_SPTR>(value.empty() ? "" : value.data()), value.size(),
        0, 0, context.m_data.get(), context.m_context.get());
    if (result >= 0)
    {
        return Success(true);
    }
    if (result != PCRE2_ERROR_NOMATCH)
    {
        const auto status = result == PCRE2_ERROR_MATCHLIMIT ||
                                    result == PCRE2_ERROR_DEPTHLIMIT ||
                                    result == PCRE2_ERROR_HEAPLIMIT
                                ? JsonSchemaValidationStatus::RegexResourceLimitExceeded
                                : JsonSchemaValidationStatus::RegexEvaluationFailed;
        return Failure(JsonSchemaRegexError{
            "regular expression matching failed: " + RegexErrorMessage(result),
            status});
    }
    return Success(false);
}

} // namespace rad::detail
