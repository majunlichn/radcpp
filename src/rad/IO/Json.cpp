#include <rad/IO/File.h>
#include <rad/IO/Json.h>
#include <rad/Core/Unicode.h>

#include <boost/json/parse.hpp>
#include <boost/json/serialize.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <optional>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
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

namespace rad
{
namespace
{

#if RAD_JSON_SCHEMA_USE_STD_REGEX
using JsonSchemaRegex = std::regex;
#else
using JsonSchemaRegex = std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)>;

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

void PrettyJsonImpl(const JsonValue& value, std::string_view indent, std::string& output,
                    std::string& currentIndent)
{
    switch (value.kind())
    {
    case JsonKind::array:
    {
        const auto& array = value.as_array();
        if (array.empty())
        {
            output += "[]";
            return;
        }

        output += "[\n";
        currentIndent += indent;
        for (std::size_t i = 0; i < array.size(); ++i)
        {
            output += currentIndent;
            PrettyJsonImpl(array[i], indent, output, currentIndent);
            if (i + 1 != array.size())
            {
                output += ',';
            }
            output += '\n';
        }
        currentIndent.resize(currentIndent.size() - indent.size());
        output += currentIndent;
        output += ']';
        return;
    }

    case JsonKind::object:
    {
        const auto& object = value.as_object();
        if (object.empty())
        {
            output += "{}";
            return;
        }

        output += "{\n";
        currentIndent += indent;
        std::size_t i = 0;
        for (const auto& member : object)
        {
            output += currentIndent;
            output += boost::json::serialize(member.key());
            output += ": ";
            PrettyJsonImpl(member.value(), indent, output, currentIndent);
            if (++i != object.size())
            {
                output += ',';
            }
            output += '\n';
        }
        currentIndent.resize(currentIndent.size() - indent.size());
        output += currentIndent;
        output += '}';
        return;
    }

    default:
        output += boost::json::serialize(value);
        return;
    }
}

[[nodiscard]] std::string_view ToStringView(const JsonString& value) noexcept
{
    return {value.data(), value.size()};
}

[[nodiscard]] std::string EscapeJsonPointerToken(std::string_view token)
{
    std::string escaped;
    escaped.reserve(token.size());
    for (const char character : token)
    {
        if (character == '~')
        {
            escaped += "~0";
        }
        else if (character == '/')
        {
            escaped += "~1";
        }
        else
        {
            escaped += character;
        }
    }
    return escaped;
}

[[nodiscard]] std::string ChildPath(std::string_view path, std::string_view token)
{
    std::string child(path);
    child += '/';
    child += EscapeJsonPointerToken(token);
    return child;
}

[[nodiscard]] int HexDigit(char character) noexcept
{
    if (character >= '0' && character <= '9')
    {
        return character - '0';
    }
    if (character >= 'a' && character <= 'f')
    {
        return character - 'a' + 10;
    }
    if (character >= 'A' && character <= 'F')
    {
        return character - 'A' + 10;
    }
    return -1;
}

[[nodiscard]] std::optional<std::string> DecodePointerFragment(std::string_view fragment)
{
    std::string pointer;
    for (std::size_t index = 0; index < fragment.size(); ++index)
    {
        if (fragment[index] == '%')
        {
            if (index + 2 >= fragment.size())
            {
                return std::nullopt;
            }
            const int high = HexDigit(fragment[index + 1]);
            const int low = HexDigit(fragment[index + 2]);
            if (high < 0 || low < 0)
            {
                return std::nullopt;
            }
            pointer += static_cast<char>((high << 4) | low);
            index += 2;
        }
        else
        {
            pointer += fragment[index];
        }
    }
    return pointer;
}

[[nodiscard]] std::optional<std::string> DecodePointerToken(std::string_view token)
{
    std::string decoded;
    for (std::size_t index = 0; index < token.size(); ++index)
    {
        if (token[index] != '~')
        {
            decoded += token[index];
            continue;
        }
        if (++index == token.size() || (token[index] != '0' && token[index] != '1'))
        {
            return std::nullopt;
        }
        decoded += token[index] == '0' ? '~' : '/';
    }
    return decoded;
}

[[nodiscard]] bool IsNumber(const JsonValue& value) noexcept
{
    return value.is_int64() || value.is_uint64() || value.is_double();
}

[[nodiscard]] long double AsNumber(const JsonValue& value) noexcept
{
    if (value.is_int64())
    {
        return static_cast<long double>(value.as_int64());
    }
    if (value.is_uint64())
    {
        return static_cast<long double>(value.as_uint64());
    }
    return static_cast<long double>(value.as_double());
}

[[nodiscard]] bool IsInteger(const JsonValue& value) noexcept
{
    if (value.is_int64() || value.is_uint64())
    {
        return true;
    }
    return value.is_double() && std::isfinite(value.as_double()) &&
           std::trunc(value.as_double()) == value.as_double();
}

[[nodiscard]] int CompareNumbers(const JsonValue& left, const JsonValue& right) noexcept
{
    if (left.is_int64() && right.is_int64())
    {
        return (left.as_int64() > right.as_int64()) -
               (left.as_int64() < right.as_int64());
    }
    if (left.is_uint64() && right.is_uint64())
    {
        return (left.as_uint64() > right.as_uint64()) -
               (left.as_uint64() < right.as_uint64());
    }
    if (left.is_int64() && right.is_uint64())
    {
        if (left.as_int64() < 0)
        {
            return -1;
        }
        const auto converted = static_cast<std::uint64_t>(left.as_int64());
        return (converted > right.as_uint64()) - (converted < right.as_uint64());
    }
    if (left.is_uint64() && right.is_int64())
    {
        return -CompareNumbers(right, left);
    }
    if (left.is_int64() && right.is_double())
    {
        const double number = right.as_double();
        constexpr double lowerBound = -9223372036854775808.0;
        constexpr double upperBound = 9223372036854775808.0;
        if (number < lowerBound)
        {
            return 1;
        }
        if (number >= upperBound)
        {
            return -1;
        }
        const auto integer = static_cast<std::int64_t>(number);
        if (left.as_int64() != integer)
        {
            return (left.as_int64() > integer) - (left.as_int64() < integer);
        }
        return (static_cast<double>(integer) > number) -
               (static_cast<double>(integer) < number);
    }
    if (left.is_uint64() && right.is_double())
    {
        const double number = right.as_double();
        constexpr double upperBound = 18446744073709551616.0;
        if (number < 0)
        {
            return 1;
        }
        if (number >= upperBound)
        {
            return -1;
        }
        const auto integer = static_cast<std::uint64_t>(number);
        if (left.as_uint64() != integer)
        {
            return (left.as_uint64() > integer) - (left.as_uint64() < integer);
        }
        return (static_cast<double>(integer) > number) -
               (static_cast<double>(integer) < number);
    }
    if (left.is_double() && !right.is_double())
    {
        return -CompareNumbers(right, left);
    }

    const auto leftNumber = AsNumber(left);
    const auto rightNumber = AsNumber(right);
    return (leftNumber > rightNumber) - (leftNumber < rightNumber);
}

[[nodiscard]] std::uint64_t IntegerMagnitude(const JsonValue& value) noexcept
{
    if (value.is_uint64())
    {
        return value.as_uint64();
    }
    const auto integer = value.as_int64();
    if (integer >= 0)
    {
        return static_cast<std::uint64_t>(integer);
    }
    return static_cast<std::uint64_t>(-(integer + 1)) + 1;
}

[[nodiscard]] std::size_t Utf8CodePointCount(std::string_view value) noexcept
{
    return static_cast<std::size_t>(std::count_if(
        value.begin(), value.end(),
        [](const unsigned char character) { return (character & 0xc0U) != 0x80U; }));
}

[[nodiscard]] std::optional<std::size_t> NonNegativeSize(const JsonValue& value) noexcept
{
    std::uint64_t size = 0;
    if (value.is_uint64())
    {
        size = value.as_uint64();
    }
    else if (value.is_int64() && value.as_int64() >= 0)
    {
        size = static_cast<std::uint64_t>(value.as_int64());
    }
    else if (value.is_double() && value.as_double() >= 0.0 &&
             std::trunc(value.as_double()) == value.as_double() &&
             value.as_double() <
                 static_cast<double>(std::numeric_limits<std::size_t>::max()))
    {
        return static_cast<std::size_t>(value.as_double());
    }
    else
    {
        return std::nullopt;
    }

    if (size > std::numeric_limits<std::size_t>::max())
    {
        return std::nullopt;
    }
    return static_cast<std::size_t>(size);
}

[[nodiscard]] bool MatchesType(const JsonValue& instance, std::string_view type) noexcept
{
    if (type == "null")
    {
        return instance.is_null();
    }
    if (type == "boolean")
    {
        return instance.is_bool();
    }
    if (type == "object")
    {
        return instance.is_object();
    }
    if (type == "array")
    {
        return instance.is_array();
    }
    if (type == "number")
    {
        return IsNumber(instance);
    }
    if (type == "integer")
    {
        return IsInteger(instance);
    }
    if (type == "string")
    {
        return instance.is_string();
    }
    return false;
}

[[nodiscard]] bool IsKnownType(std::string_view type) noexcept
{
    constexpr std::array types = {"null", "boolean", "object", "array",
                                  "number", "integer", "string"};
    return std::find(types.begin(), types.end(), type) != types.end();
}

[[nodiscard]] bool JsonSchemaEqual(const JsonValue& lhs, const JsonValue& rhs)
{
    if (IsNumber(lhs) && IsNumber(rhs))
    {
        return CompareNumbers(lhs, rhs) == 0;
    }
    if (lhs.kind() != rhs.kind())
    {
        return false;
    }
    if (lhs.is_array())
    {
        const auto& lhsArray = lhs.as_array();
        const auto& rhsArray = rhs.as_array();
        if (lhsArray.size() != rhsArray.size())
        {
            return false;
        }
        for (std::size_t index = 0; index < lhsArray.size(); ++index)
        {
            if (!JsonSchemaEqual(lhsArray[index], rhsArray[index]))
            {
                return false;
            }
        }
        return true;
    }
    if (lhs.is_object())
    {
        const auto& lhsObject = lhs.as_object();
        const auto& rhsObject = rhs.as_object();
        if (lhsObject.size() != rhsObject.size())
        {
            return false;
        }
        for (const auto& member : lhsObject)
        {
            const auto* rhsValue = rhsObject.if_contains(member.key());
            if (rhsValue == nullptr || !JsonSchemaEqual(member.value(), *rhsValue))
            {
                return false;
            }
        }
        return true;
    }
    return lhs == rhs;
}

[[nodiscard]] bool HasDuplicates(const JsonArray& values)
{
    for (std::size_t first = 0; first < values.size(); ++first)
    {
        for (std::size_t second = first + 1; second < values.size(); ++second)
        {
            if (JsonSchemaEqual(values[first], values[second]))
            {
                return true;
            }
        }
    }
    return false;
}

class JsonSchemaValidator
{
public:
    JsonSchemaValidator(JsonSchemaDialect dialect,
                        const JsonSchemaValidationOptions& options,
                        const JsonValue& rootSchema)
        : m_dialect(dialect), m_options(options), m_rootSchema(rootSchema)
    {
    }

    [[nodiscard]] std::optional<JsonSchemaCompileError>
    CheckSchema(const JsonValue& schema)
    {
        m_checkingSchema = true;
        ValidateSchemaDefinition(schema, {}, 0);
        return std::move(m_compileError);
    }

    [[nodiscard]] JsonSchemaValidationResult ValidateInstance(const JsonValue& schema,
                                                              const JsonValue& instance)
    {
        Validate(schema, instance, {}, {}, 0);
        return std::move(m_result);
    }

private:
    struct ReferenceTarget
    {
        const JsonValue* schema;
        std::string path;
    };

    [[nodiscard]] bool IsEmbeddedResourcePath(std::string_view path) const
    {
        const JsonValue* value = &m_rootSchema;
        for (std::size_t begin = 1; value != nullptr;)
        {
            if (value != &m_rootSchema && value->is_object() &&
                value->as_object().contains("$id"))
            {
                return true;
            }
            if (begin > path.size())
            {
                break;
            }
            const auto end = path.find('/', begin);
            const auto token = DecodePointerToken(path.substr(
                begin, end == std::string_view::npos ? end : end - begin));
            if (!token)
            {
                break;
            }
            if (value->is_object())
            {
                value = value->as_object().if_contains(*token);
            }
            else if (value->is_array())
            {
                std::size_t index = 0;
                const auto [next, error] = std::from_chars(
                    token->data(), token->data() + token->size(), index);
                value = error == std::errc{} && next == token->data() + token->size() &&
                                index < value->as_array().size()
                            ? &value->as_array()[index]
                            : nullptr;
            }
            else
            {
                value = nullptr;
            }
            if (end == std::string_view::npos)
            {
                break;
            }
            begin = end + 1;
        }
        return value != nullptr && value != &m_rootSchema && value->is_object() &&
               value->as_object().contains("$id");
    }

    [[nodiscard]] std::optional<ReferenceTarget>
    ResolveReference(const JsonValue& reference, std::string_view schemaPath)
    {
        if (!reference.is_string())
        {
            AddError({}, schemaPath, "$ref must be a string");
            return std::nullopt;
        }
        if (IsEmbeddedResourcePath(schemaPath.substr(0, schemaPath.size() - 5)))
        {
            AddUnsupportedError(
                {}, schemaPath,
                "references inside embedded $id resources are not supported");
            return std::nullopt;
        }
        const auto uri = ToStringView(reference.as_string());
        if (uri.empty() || uri.front() != '#')
        {
            AddUnsupportedError(
                {}, schemaPath, "only root-local JSON Pointer references are supported");
            return std::nullopt;
        }
        if (uri.size() > 1 && uri[1] != '/' && uri[1] != '%')
        {
            AddUnsupportedError({}, schemaPath, "anchor references are not supported");
            return std::nullopt;
        }
        const auto pointer = DecodePointerFragment(uri.substr(1));
        if (!pointer)
        {
            AddError({}, schemaPath, "invalid percent escape in $ref");
            return std::nullopt;
        }
        if (!pointer->empty() && pointer->front() != '/')
        {
            AddUnsupportedError({}, schemaPath, "anchor references are not supported");
            return std::nullopt;
        }

        const JsonValue* target = &m_rootSchema;
        std::string targetPath;
        for (std::size_t begin = 1; begin <= pointer->size();)
        {
            const auto end = pointer->find('/', begin);
            const auto token = DecodePointerToken(std::string_view(*pointer).substr(
                begin, end == std::string::npos ? end : end - begin));
            if (!token)
            {
                AddError({}, schemaPath, "invalid JSON Pointer escape in $ref");
                return std::nullopt;
            }
            targetPath = ChildPath(targetPath, *token);
            if (target->is_object())
            {
                target = target->as_object().if_contains(*token);
            }
            else if (target->is_array())
            {
                const auto& array = target->as_array();
                std::size_t index = 0;
                if (token->empty() || (token->size() > 1 && token->front() == '0'))
                {
                    target = nullptr;
                }
                else
                {
                    for (const char digit : *token)
                    {
                        if (digit < '0' || digit > '9' ||
                            index > (std::numeric_limits<std::size_t>::max() -
                                     static_cast<std::size_t>(digit - '0')) / 10)
                        {
                            target = nullptr;
                            break;
                        }
                        index = index * 10 + static_cast<std::size_t>(digit - '0');
                    }
                    if (target != nullptr)
                    {
                        target = index < array.size() ? &array[index] : nullptr;
                    }
                }
            }
            else
            {
                target = nullptr;
            }
            if (target == nullptr)
            {
                AddError({}, schemaPath, "$ref target does not exist");
                return std::nullopt;
            }
            if (end == std::string::npos)
            {
                break;
            }
            begin = end + 1;
        }
        if (IsEmbeddedResourcePath(targetPath))
        {
            AddUnsupportedError(
                {}, schemaPath,
                "references into embedded $id resources are not supported");
            return std::nullopt;
        }
        return ReferenceTarget{target, std::move(targetPath)};
    }

    void AddError(std::string_view instancePath, std::string_view schemaPath,
                  std::string message)
    {
        if (m_checkingSchema)
        {
            if (!m_compileError)
            {
                m_compileError = JsonSchemaCompileError{
                    JsonSchemaCompileErrorCode::InvalidSchema,
                    m_dialect,
                    std::string(schemaPath),
                    std::move(message),
                };
            }
            return;
        }
        if (m_result.errors.size() >= std::max<std::size_t>(m_options.maxErrors, 1))
        {
            return;
        }
        m_result.errors.push_back(
            {std::string(instancePath), std::string(schemaPath), std::move(message)});
    }

    void AddUnsupportedError(std::string_view instancePath, std::string_view schemaPath,
                             std::string message)
    {
        if (m_checkingSchema && !m_compileError)
        {
            m_compileError = JsonSchemaCompileError{
                JsonSchemaCompileErrorCode::UnsupportedFeature,
                m_dialect,
                std::string(schemaPath),
                std::move(message),
            };
        }
        if (m_checkingSchema)
        {
            return;
        }
        AddError(instancePath, schemaPath, std::move(message));
    }

    void AddResourceError(std::string_view instancePath, std::string_view schemaPath,
                          std::string message)
    {
        m_resourceError = true;
        AddError(instancePath, schemaPath, std::move(message));
    }

    [[nodiscard]] std::optional<JsonSchemaRegex>
    CompilePattern(std::string_view pattern, std::string_view instancePath,
                   std::string_view schemaPath)
    {
#if RAD_JSON_SCHEMA_USE_STD_REGEX
        if (pattern.find("\\p{") != std::string_view::npos ||
            pattern.find("\\P{") != std::string_view::npos)
        {
            AddUnsupportedError(instancePath, schemaPath,
                                "Unicode property escapes are not supported by std::regex");
            return std::nullopt;
        }
        try
        {
            return std::regex(std::string(pattern));
        }
        catch (const std::regex_error&)
        {
            AddError(instancePath, schemaPath, "pattern is not a valid regular expression");
            return std::nullopt;
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
            AddError(instancePath, schemaPath,
                     "whitespace escape cannot be a character range endpoint");
            return std::nullopt;
        }
        JsonSchemaRegex expression(
            pcre2_compile(reinterpret_cast<PCRE2_SPTR>(normalized->data()),
                          normalized->size(), options, &errorCode, &errorOffset, nullptr),
            &pcre2_code_free);
        if (!expression)
        {
            AddError(instancePath, schemaPath,
                     "pattern is not a valid regular expression at normalized byte " +
                         std::to_string(errorOffset) + ": " + RegexErrorMessage(errorCode));
            return std::nullopt;
        }
        return expression;
#endif
    }

    [[nodiscard]] bool MatchesPattern(const JsonSchemaRegex& expression, std::string_view value,
                                      std::string_view instancePath,
                                      std::string_view schemaPath)
    {
#if RAD_JSON_SCHEMA_USE_STD_REGEX
        try
        {
            return std::regex_search(value.begin(), value.end(), expression);
        }
        catch (const std::regex_error& error)
        {
            AddResourceError(instancePath, schemaPath,
                             std::string("regular expression matching failed: ") + error.what());
            return false;
        }
#else
        const std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> matchData(
            pcre2_match_data_create_from_pattern(expression.get(), nullptr),
            &pcre2_match_data_free);
        const std::unique_ptr<pcre2_match_context, decltype(&pcre2_match_context_free)> context(
            pcre2_match_context_create(nullptr), &pcre2_match_context_free);
        if (!matchData || !context)
        {
            AddResourceError(instancePath, schemaPath,
                             "unable to allocate regular expression matching state");
            return false;
        }
        if (pcre2_set_match_limit(context.get(), 1000000) != 0 ||
            pcre2_set_depth_limit(context.get(), 1000) != 0 ||
            pcre2_set_heap_limit(context.get(), 8192) != 0)
        {
            AddResourceError(instancePath, schemaPath,
                             "unable to set regular expression matching limits");
            return false;
        }
        const int result = pcre2_match(
            expression.get(),
            reinterpret_cast<PCRE2_SPTR>(value.empty() ? "" : value.data()), value.size(),
            0, 0, matchData.get(), context.get());
        if (result >= 0)
        {
            return true;
        }
        if (result != PCRE2_ERROR_NOMATCH)
        {
            AddResourceError(instancePath, schemaPath,
                             "regular expression matching failed: " + RegexErrorMessage(result));
        }
        return false;
#endif
    }

    void ValidateSchemaDefinition(const JsonValue& schema, std::string_view schemaPath,
                                  std::size_t depth)
    {
        if (m_compileError)
        {
            return;
        }
        if (depth > m_options.maxDepth)
        {
            AddError({}, schemaPath, "maximum schema depth exceeded");
            return;
        }
        if (schema.is_bool())
        {
            return;
        }
        if (!schema.is_object())
        {
            AddError({}, schemaPath, "schema must be an object or boolean");
            return;
        }
        if (std::find(m_checkedSchemas.begin(), m_checkedSchemas.end(), &schema) !=
            m_checkedSchemas.end())
        {
            return;
        }
        m_checkedSchemas.push_back(&schema);

        const auto& object = schema.as_object();
        if (const auto* reference = object.if_contains("$ref"))
        {
            const auto target = ResolveReference(*reference, ChildPath(schemaPath, "$ref"));
            if (!target)
            {
                return;
            }
            ValidateSchemaDefinition(*target->schema, target->path, depth + 1);
            if (m_dialect == JsonSchemaDialect::Draft7)
            {
                return;
            }
        }
        ValidateUnsupportedKeywords(object, {}, schemaPath);

        if (const auto* declaredDialect = object.if_contains("$schema"))
        {
            const auto keywordPath = ChildPath(schemaPath, "$schema");
            if (!declaredDialect->is_string())
            {
                AddError({}, keywordPath, "$schema must be a string");
            }
            else
            {
                std::string_view expected;
                switch (m_dialect)
                {
                case JsonSchemaDialect::Draft7:
                    expected = "http://json-schema.org/draft-07/schema";
                    break;
                case JsonSchemaDialect::Draft2019_09:
                    expected = "https://json-schema.org/draft/2019-09/schema";
                    break;
                case JsonSchemaDialect::Draft2020_12:
                    expected = "https://json-schema.org/draft/2020-12/schema";
                    break;
                }
                auto declared = ToStringView(declaredDialect->as_string());
                if (declared.ends_with('#'))
                {
                    declared.remove_suffix(1);
                }
                if (declared != expected)
                {
                    AddUnsupportedError(
                        {}, keywordPath,
                        "custom or mismatched meta-schemas are not supported");
                }
            }
        }

        if (const auto* type = object.if_contains("type"))
        {
            bool valid = false;
            if (type->is_string())
            {
                valid = IsKnownType(ToStringView(type->as_string()));
            }
            else if (type->is_array() && !type->as_array().empty())
            {
                valid = std::all_of(
                    type->as_array().begin(), type->as_array().end(),
                    [](const JsonValue& candidate) {
                        return candidate.is_string() &&
                               IsKnownType(ToStringView(candidate.as_string()));
                    }) &&
                        !HasDuplicates(type->as_array());
            }
            if (!valid)
            {
                AddError({}, ChildPath(schemaPath, "type"),
                         "type must be a known type name or a non-empty array of type names");
            }
        }

        if (const auto* enumeration = object.if_contains("enum");
            enumeration != nullptr &&
            (!enumeration->is_array() || enumeration->as_array().empty() ||
             HasDuplicates(enumeration->as_array())))
        {
            AddError({}, ChildPath(schemaPath, "enum"),
                     "enum must be a non-empty array of unique values");
        }

        constexpr std::array sizeKeywords = {
            "minProperties", "maxProperties", "minItems",
            "maxItems",      "minLength",     "maxLength",
        };
        for (const std::string_view keyword : sizeKeywords)
        {
            if (const auto* value = object.if_contains(keyword);
                value != nullptr && !NonNegativeSize(*value))
            {
                AddError({}, ChildPath(schemaPath, keyword),
                         std::string(keyword) + " must be a non-negative integer");
            }
        }
        if (m_dialect != JsonSchemaDialect::Draft7)
        {
            constexpr std::array containsSizeKeywords = {"minContains", "maxContains"};
            for (const std::string_view keyword : containsSizeKeywords)
            {
                if (const auto* value = object.if_contains(keyword);
                    value != nullptr && !NonNegativeSize(*value))
                {
                    AddError({}, ChildPath(schemaPath, keyword),
                             std::string(keyword) + " must be a non-negative integer");
                }
            }
        }

        constexpr std::array numberKeywords = {
            "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        };
        for (const std::string_view keyword : numberKeywords)
        {
            if (const auto* value = object.if_contains(keyword);
                value != nullptr &&
                (!IsNumber(*value) ||
                 (value->is_double() && !std::isfinite(value->as_double()))))
            {
                AddError({}, ChildPath(schemaPath, keyword),
                         std::string(keyword) + " must be a finite number");
            }
        }
        if (const auto* value = object.if_contains("multipleOf"))
        {
            const auto keywordPath = ChildPath(schemaPath, "multipleOf");
            if (!IsNumber(*value) ||
                (value->is_double() && !std::isfinite(value->as_double())) ||
                AsNumber(*value) <= 0)
            {
                AddError({}, keywordPath, "multipleOf must be a positive finite number");
            }
            else if (value->is_double() && !IsInteger(*value))
            {
                AddUnsupportedError(
                    {}, keywordPath,
                    "fractional multipleOf is not supported by this validator");
            }
        }

        if (const auto* pattern = object.if_contains("pattern"))
        {
            const auto patternPath = ChildPath(schemaPath, "pattern");
            if (!pattern->is_string())
            {
                AddError({}, patternPath, "pattern must be a string");
            }
            else
            {
                static_cast<void>(CompilePattern(ToStringView(pattern->as_string()), {},
                                                 patternPath));
            }
        }

        if (const auto* required = object.if_contains("required"))
        {
            const auto requiredPath = ChildPath(schemaPath, "required");
            if (!required->is_array())
            {
                AddError({}, requiredPath, "required must be an array of strings");
            }
            else
            {
                if (HasDuplicates(required->as_array()))
                {
                    AddError({}, requiredPath,
                             "required property names must be unique");
                }
                for (std::size_t index = 0; index < required->as_array().size(); ++index)
                {
                    if (!required->as_array()[index].is_string())
                    {
                        AddError({}, ChildPath(requiredPath, std::to_string(index)),
                                 "required property name must be a string");
                    }
                }
            }
        }

        if (const auto* unique = object.if_contains("uniqueItems");
            unique != nullptr && !unique->is_bool())
        {
            AddError({}, ChildPath(schemaPath, "uniqueItems"),
                     "uniqueItems must be a boolean");
        }

        if (const auto* properties = object.if_contains("properties"))
        {
            const auto propertiesPath = ChildPath(schemaPath, "properties");
            if (!properties->is_object())
            {
                AddError({}, propertiesPath, "properties must be an object");
            }
            else
            {
                for (const auto& property : properties->as_object())
                {
                    ValidateSchemaDefinition(property.value(),
                                             ChildPath(propertiesPath, property.key()),
                                             depth + 1);
                }
            }
        }

        if (const auto* patterns = object.if_contains("patternProperties"))
        {
            const auto keywordPath = ChildPath(schemaPath, "patternProperties");
            if (!patterns->is_object())
            {
                AddError({}, keywordPath, "patternProperties must be an object");
            }
            else
            {
                for (const auto& pattern : patterns->as_object())
                {
                    const auto patternPath = ChildPath(keywordPath, pattern.key());
                    static_cast<void>(CompilePattern(pattern.key(), {}, patternPath));
                    ValidateSchemaDefinition(pattern.value(), patternPath, depth + 1);
                }
            }
        }
        if (const auto* names = object.if_contains("propertyNames"))
        {
            ValidateSchemaDefinition(*names, ChildPath(schemaPath, "propertyNames"),
                                     depth + 1);
        }

        const std::string_view definitionsKeyword =
            m_dialect == JsonSchemaDialect::Draft7 ? "definitions" : "$defs";
        if (const auto* definitions = object.if_contains(definitionsKeyword))
        {
            const auto definitionsPath = ChildPath(schemaPath, definitionsKeyword);
            if (!definitions->is_object())
            {
                AddError({}, definitionsPath,
                         std::string(definitionsKeyword) + " must be an object");
            }
            else
            {
                for (const auto& definition : definitions->as_object())
                {
                    ValidateSchemaDefinition(
                        definition.value(), ChildPath(definitionsPath, definition.key()),
                        depth + 1);
                }
            }
        }

        ValidateDependenciesSchema(object, schemaPath, depth);

        if (const auto* contains = object.if_contains("contains"))
        {
            ValidateSchemaDefinition(*contains, ChildPath(schemaPath, "contains"),
                                     depth + 1);
        }
        constexpr std::array conditionalKeywords = {"if", "then", "else"};
        for (const std::string_view keyword : conditionalKeywords)
        {
            if (const auto* conditional = object.if_contains(keyword))
            {
                ValidateSchemaDefinition(*conditional, ChildPath(schemaPath, keyword),
                                         depth + 1);
            }
        }

        if (m_dialect == JsonSchemaDialect::Draft2020_12)
        {
            if (const auto* prefixItems = object.if_contains("prefixItems"))
            {
                const auto prefixPath = ChildPath(schemaPath, "prefixItems");
                if (!prefixItems->is_array())
                {
                    AddError({}, prefixPath, "prefixItems must be an array");
                }
                else
                {
                    for (std::size_t index = 0;
                         index < prefixItems->as_array().size(); ++index)
                    {
                        ValidateSchemaDefinition(
                            prefixItems->as_array()[index],
                            ChildPath(prefixPath, std::to_string(index)), depth + 1);
                    }
                }
            }
        }

        if (const auto* additional = object.if_contains("additionalProperties"))
        {
            const auto additionalPath = ChildPath(schemaPath, "additionalProperties");
            if (!additional->is_bool() && !additional->is_object())
            {
                AddError({}, additionalPath,
                         "additionalProperties must be a boolean or schema");
            }
            else
            {
                ValidateSchemaDefinition(*additional, additionalPath, depth + 1);
            }
        }

        if (const auto* items = object.if_contains("items"))
        {
            const auto itemsPath = ChildPath(schemaPath, "items");
            if (items->is_array() && m_dialect != JsonSchemaDialect::Draft2020_12)
            {
                if (items->as_array().empty())
                {
                    AddError({}, itemsPath, "tuple items must be a non-empty array of schemas");
                }
                for (std::size_t index = 0; index < items->as_array().size(); ++index)
                {
                    ValidateSchemaDefinition(
                        items->as_array()[index],
                        ChildPath(itemsPath, std::to_string(index)), depth + 1);
                }
            }
            else
            {
                ValidateSchemaDefinition(*items, itemsPath, depth + 1);
            }
        }
        if (m_dialect != JsonSchemaDialect::Draft2020_12)
        {
            if (const auto* additional = object.if_contains("additionalItems"))
            {
                ValidateSchemaDefinition(*additional, ChildPath(schemaPath, "additionalItems"),
                                         depth + 1);
            }
        }

        constexpr std::array compositionKeywords = {"allOf", "anyOf", "oneOf"};
        for (const std::string_view keyword : compositionKeywords)
        {
            if (const auto* alternatives = object.if_contains(keyword))
            {
                const auto keywordPath = ChildPath(schemaPath, keyword);
                if (!alternatives->is_array() || alternatives->as_array().empty())
                {
                    AddError({}, keywordPath,
                             std::string(keyword) + " must be a non-empty array");
                }
                else
                {
                    for (std::size_t index = 0; index < alternatives->as_array().size();
                         ++index)
                    {
                        ValidateSchemaDefinition(
                            alternatives->as_array()[index],
                            ChildPath(keywordPath, std::to_string(index)), depth + 1);
                    }
                }
            }
        }
        if (const auto* negated = object.if_contains("not"))
        {
            ValidateSchemaDefinition(*negated, ChildPath(schemaPath, "not"), depth + 1);
        }
    }

    void ValidateDependenciesSchema(const JsonObject& schema,
                                    std::string_view schemaPath, std::size_t depth)
    {
        constexpr std::array keywords = {
            "dependencies", "dependentRequired", "dependentSchemas",
        };
        for (const std::string_view keyword : keywords)
        {
            if ((keyword == "dependencies") != (m_dialect == JsonSchemaDialect::Draft7))
            {
                continue;
            }
            const auto* dependencies = schema.if_contains(keyword);
            if (dependencies == nullptr)
            {
                continue;
            }
            const auto keywordPath = ChildPath(schemaPath, keyword);
            if (!dependencies->is_object())
            {
                AddError({}, keywordPath, std::string(keyword) + " must be an object");
                continue;
            }
            for (const auto& dependency : dependencies->as_object())
            {
                const auto dependencyPath = ChildPath(keywordPath, dependency.key());
                const auto& value = dependency.value();
                if (keyword == "dependentRequired" ||
                    (keyword == "dependencies" && value.is_array()))
                {
                    if (!value.is_array())
                    {
                        AddError({}, dependencyPath,
                                 "dependentRequired value must be an array of strings");
                        continue;
                    }
                    if (HasDuplicates(value.as_array()))
                    {
                        AddError({}, dependencyPath,
                                 "dependent property names must be unique");
                    }
                    for (std::size_t index = 0;
                         index < value.as_array().size(); ++index)
                    {
                        if (!value.as_array()[index].is_string())
                        {
                            AddError({}, ChildPath(dependencyPath, std::to_string(index)),
                                     "dependent property name must be a string");
                        }
                    }
                }
                else
                {
                    ValidateSchemaDefinition(value, dependencyPath, depth + 1);
                }
            }
        }
    }

    [[nodiscard]] std::optional<bool>
    BranchMatches(const JsonValue& schema, const JsonValue& instance,
                  std::string_view instancePath, std::string_view schemaPath,
                  std::size_t depth)
    {
        JsonSchemaValidationOptions options = m_options;
        options.maxErrors = 1;
        JsonSchemaValidator validator(m_dialect, options, m_rootSchema);
        validator.Validate(schema, instance, instancePath, schemaPath, depth);
        if (validator.m_resourceError)
        {
            m_resourceError = true;
            for (auto& error : validator.m_result.errors)
            {
                AddError(error.instancePath, error.schemaPath, std::move(error.message));
            }
            return std::nullopt;
        }
        return static_cast<bool>(validator.m_result);
    }

    void Validate(const JsonValue& schema, const JsonValue& instance,
                  std::string_view instancePath, std::string_view schemaPath,
                  std::size_t depth)
    {
        if (m_result.errors.size() >= std::max<std::size_t>(m_options.maxErrors, 1))
        {
            return;
        }
        if (depth > m_options.maxDepth)
        {
            AddResourceError(instancePath, schemaPath,
                             "maximum validation depth exceeded");
            return;
        }
        if (schema.is_bool())
        {
            if (!schema.as_bool())
            {
                AddError(instancePath, schemaPath, "value is rejected by the false schema");
            }
            return;
        }
        if (!schema.is_object())
        {
            AddError(instancePath, schemaPath, "schema must be an object or boolean");
            return;
        }

        const auto& object = schema.as_object();
        if (const auto* reference = object.if_contains("$ref"))
        {
            const auto target = ResolveReference(*reference, ChildPath(schemaPath, "$ref"));
            if (!target)
            {
                return;
            }
            Validate(*target->schema, instance, instancePath, target->path, depth + 1);
            if (m_dialect == JsonSchemaDialect::Draft7)
            {
                return;
            }
        }
        ValidateUnsupportedKeywords(object, instancePath, schemaPath);
        ValidateType(object, instance, instancePath, schemaPath);
        ValidateEnumAndConst(object, instance, instancePath, schemaPath);
        ValidateCompositions(object, instance, instancePath, schemaPath, depth);

        if (instance.is_object())
        {
            ValidateObject(object, instance, instancePath, schemaPath, depth);
        }
        if (instance.is_array())
        {
            ValidateArray(object, instance.as_array(), instancePath, schemaPath, depth);
        }
        if (instance.is_string())
        {
            ValidateString(object, instance.as_string(), instancePath, schemaPath);
        }
        if (IsNumber(instance))
        {
            ValidateNumber(object, instance, instancePath, schemaPath);
        }
    }

    void ValidateUnsupportedKeywords(const JsonObject& schema, std::string_view instancePath,
                                     std::string_view schemaPath)
    {
        if (m_dialect != JsonSchemaDialect::Draft7 && schema.contains("$vocabulary"))
        {
            AddUnsupportedError(instancePath, ChildPath(schemaPath, "$vocabulary"),
                                "custom vocabularies are not supported by this validator");
        }

        if (m_dialect != JsonSchemaDialect::Draft7)
        {
            constexpr std::array newerUnsupported = {
                "unevaluatedProperties",
                "unevaluatedItems",
            };
            for (const std::string_view keyword : newerUnsupported)
            {
                if (schema.contains(keyword))
                {
                    AddUnsupportedError(instancePath, ChildPath(schemaPath, keyword),
                                        "keyword is not supported by this validator");
                }
            }
        }

        if (m_dialect == JsonSchemaDialect::Draft2019_09)
        {
            constexpr std::array recursiveReferences = {"$recursiveRef", "$recursiveAnchor"};
            for (const std::string_view keyword : recursiveReferences)
            {
                if (schema.contains(keyword))
                {
                    AddUnsupportedError(instancePath, ChildPath(schemaPath, keyword),
                                        "keyword is not supported by this validator");
                }
            }
        }
        else if (m_dialect == JsonSchemaDialect::Draft2020_12)
        {
            constexpr std::array dynamicReferences = {"$dynamicRef", "$dynamicAnchor"};
            for (const std::string_view keyword : dynamicReferences)
            {
                if (schema.contains(keyword))
                {
                    AddUnsupportedError(instancePath, ChildPath(schemaPath, keyword),
                                        "keyword is not supported by this validator");
                }
            }
        }
    }

    void ValidateType(const JsonObject& schema, const JsonValue& instance,
                      std::string_view instancePath, std::string_view schemaPath)
    {
        const auto* typeValue = schema.if_contains("type");
        if (typeValue == nullptr)
        {
            return;
        }

        bool matches = false;
        bool validSchema = true;
        if (typeValue->is_string())
        {
            const auto type = ToStringView(typeValue->as_string());
            validSchema = IsKnownType(type);
            matches = validSchema && MatchesType(instance, type);
        }
        else if (typeValue->is_array() && !typeValue->as_array().empty())
        {
            for (const auto& candidate : typeValue->as_array())
            {
                if (!candidate.is_string() ||
                    !IsKnownType(ToStringView(candidate.as_string())))
                {
                    validSchema = false;
                    break;
                }
                matches =
                    matches || MatchesType(instance, ToStringView(candidate.as_string()));
            }
        }
        else
        {
            validSchema = false;
        }

        const auto typePath = ChildPath(schemaPath, "type");
        if (!validSchema)
        {
            AddError(instancePath, typePath,
                     "type must be a known type name or a non-empty array of type names");
        }
        else if (!matches)
        {
            AddError(instancePath, typePath, "value does not match the required type");
        }
    }

    void ValidateEnumAndConst(const JsonObject& schema, const JsonValue& instance,
                              std::string_view instancePath, std::string_view schemaPath)
    {
        if (const auto* enumValue = schema.if_contains("enum"))
        {
            if (!enumValue->is_array() || enumValue->as_array().empty())
            {
                AddError(instancePath, ChildPath(schemaPath, "enum"),
                         "enum must be a non-empty array");
            }
            else if (std::none_of(enumValue->as_array().begin(), enumValue->as_array().end(),
                                  [&instance](const JsonValue& candidate) {
                                      return JsonSchemaEqual(candidate, instance);
                                  }))
            {
                AddError(instancePath, ChildPath(schemaPath, "enum"),
                         "value is not one of the allowed values");
            }
        }
        if (const auto* constValue = schema.if_contains("const");
            constValue != nullptr && !JsonSchemaEqual(*constValue, instance))
        {
            AddError(instancePath, ChildPath(schemaPath, "const"),
                     "value does not equal the required constant");
        }
    }

    void ValidateCompositions(const JsonObject& schema, const JsonValue& instance,
                              std::string_view instancePath, std::string_view schemaPath,
                              std::size_t depth)
    {
        if (const auto* allOf = schema.if_contains("allOf"))
        {
            const auto keywordPath = ChildPath(schemaPath, "allOf");
            if (!allOf->is_array() || allOf->as_array().empty())
            {
                AddError(instancePath, keywordPath, "allOf must be a non-empty array");
            }
            else
            {
                for (std::size_t index = 0; index < allOf->as_array().size(); ++index)
                {
                    Validate(allOf->as_array()[index], instance, instancePath,
                             ChildPath(keywordPath, std::to_string(index)), depth + 1);
                }
            }
        }

        ValidateAlternative(schema, "anyOf", instance, instancePath, schemaPath, depth, false);
        ValidateAlternative(schema, "oneOf", instance, instancePath, schemaPath, depth, true);

        if (const auto* notSchema = schema.if_contains("not"))
        {
            const auto keywordPath = ChildPath(schemaPath, "not");
            const auto matches =
                BranchMatches(*notSchema, instance, instancePath, keywordPath, depth + 1);
            if (!matches)
            {
                return;
            }
            if (*matches)
            {
                AddError(instancePath, keywordPath, "value matches the disallowed schema");
            }
        }

        if (const auto* condition = schema.if_contains("if"))
        {
            const auto matches =
                BranchMatches(*condition, instance, instancePath,
                              ChildPath(schemaPath, "if"), depth + 1);
            if (!matches)
            {
                return;
            }
            const std::string_view keyword = *matches ? "then" : "else";
            if (const auto* branch = schema.if_contains(keyword))
            {
                Validate(*branch, instance, instancePath,
                         ChildPath(schemaPath, keyword), depth + 1);
            }
        }
    }

    void ValidateAlternative(const JsonObject& schema, std::string_view keyword,
                             const JsonValue& instance, std::string_view instancePath,
                             std::string_view schemaPath, std::size_t depth, bool exactlyOne)
    {
        const auto* alternatives = schema.if_contains(keyword);
        if (alternatives == nullptr)
        {
            return;
        }

        const auto keywordPath = ChildPath(schemaPath, keyword);
        if (!alternatives->is_array() || alternatives->as_array().empty())
        {
            AddError(instancePath, keywordPath,
                     std::string(keyword) + " must be a non-empty array");
            return;
        }

        std::size_t matches = 0;
        for (std::size_t index = 0; index < alternatives->as_array().size(); ++index)
        {
            const auto branchMatches =
                BranchMatches(alternatives->as_array()[index], instance, instancePath,
                              ChildPath(keywordPath, std::to_string(index)), depth + 1);
            if (!branchMatches)
            {
                return;
            }
            if (*branchMatches)
            {
                ++matches;
            }
        }
        if ((!exactlyOne && matches == 0) || (exactlyOne && matches != 1))
        {
            AddError(instancePath, keywordPath,
                     exactlyOne ? "value must match exactly one schema"
                                : "value must match at least one schema");
        }
    }

    void ValidateObject(const JsonObject& schema, const JsonValue& instanceValue,
                        std::string_view instancePath, std::string_view schemaPath,
                        std::size_t depth)
    {
        const auto& instance = instanceValue.as_object();
        ValidateSizeKeyword(schema, "minProperties", instance.size(), true, instancePath,
                            schemaPath);
        ValidateSizeKeyword(schema, "maxProperties", instance.size(), false, instancePath,
                            schemaPath);

        if (const auto* required = schema.if_contains("required"))
        {
            const auto keywordPath = ChildPath(schemaPath, "required");
            if (!required->is_array())
            {
                AddError(instancePath, keywordPath, "required must be an array of strings");
            }
            else
            {
                for (std::size_t index = 0; index < required->as_array().size(); ++index)
                {
                    const auto& name = required->as_array()[index];
                    if (!name.is_string())
                    {
                        AddError(instancePath, ChildPath(keywordPath, std::to_string(index)),
                                 "required property name must be a string");
                    }
                    else if (!instance.contains(ToStringView(name.as_string())))
                    {
                        AddError(ChildPath(instancePath, ToStringView(name.as_string())),
                                 keywordPath, "required property is missing");
                    }
                }
            }
        }

        const JsonObject* properties = nullptr;
        if (const auto* propertiesValue = schema.if_contains("properties"))
        {
            if (!propertiesValue->is_object())
            {
                AddError(instancePath, ChildPath(schemaPath, "properties"),
                         "properties must be an object");
            }
            else
            {
                properties = &propertiesValue->as_object();
                for (const auto& property : *properties)
                {
                    if (const auto* value = instance.if_contains(property.key()))
                    {
                        Validate(property.value(), *value,
                                 ChildPath(instancePath, property.key()),
                                 ChildPath(ChildPath(schemaPath, "properties"), property.key()),
                                 depth + 1);
                    }
                }
            }
        }

        std::vector<bool> patternMatches;
        if (const auto* patterns = schema.if_contains("patternProperties"))
        {
            patternMatches.resize(instance.size(), false);
            const auto keywordPath = ChildPath(schemaPath, "patternProperties");
            if (!patterns->is_object())
            {
                AddError(instancePath, keywordPath, "patternProperties must be an object");
            }
            else
            {
                for (const auto& pattern : patterns->as_object())
                {
                    const auto patternPath = ChildPath(keywordPath, pattern.key());
                    const auto expression = CompilePattern(pattern.key(), instancePath,
                                                           patternPath);
                    if (!expression)
                    {
                        continue;
                    }
                    std::size_t index = 0;
                    for (const auto& property : instance)
                    {
                        const auto name = property.key();
                        if (MatchesPattern(*expression, name, instancePath, patternPath))
                        {
                            patternMatches[index] = true;
                            Validate(pattern.value(), property.value(),
                                     ChildPath(instancePath, name), patternPath, depth + 1);
                        }
                        ++index;
                    }
                }
            }
        }

        if (const auto* names = schema.if_contains("propertyNames"))
        {
            const auto keywordPath = ChildPath(schemaPath, "propertyNames");
            for (const auto& property : instance)
            {
                Validate(*names, JsonValue(property.key()),
                         ChildPath(instancePath, property.key()), keywordPath, depth + 1);
            }
        }

        if (const auto* additional = schema.if_contains("additionalProperties"))
        {
            const auto keywordPath = ChildPath(schemaPath, "additionalProperties");
            if (!additional->is_bool() && !additional->is_object())
            {
                AddError(instancePath, keywordPath,
                         "additionalProperties must be a boolean or schema");
                return;
            }
            std::size_t index = 0;
            for (const auto& property : instance)
            {
                const bool matched = !patternMatches.empty() && patternMatches[index];
                ++index;
                if (matched || (properties != nullptr && properties->contains(property.key())))
                {
                    continue;
                }
                const auto propertyPath = ChildPath(instancePath, property.key());
                if (additional->is_bool())
                {
                    if (!additional->as_bool())
                    {
                        AddError(propertyPath, keywordPath,
                                 "additional property is not allowed");
                    }
                }
                else
                {
                    Validate(*additional, property.value(), propertyPath, keywordPath,
                             depth + 1);
                }
            }
        }

        ValidateDependencies(schema, instanceValue, instancePath, schemaPath, depth);
    }

    void ValidateDependencies(const JsonObject& schema, const JsonValue& instanceValue,
                              std::string_view instancePath, std::string_view schemaPath,
                              std::size_t depth)
    {
        const auto& instance = instanceValue.as_object();
        constexpr std::array keywords = {
            "dependencies", "dependentRequired", "dependentSchemas",
        };
        for (const std::string_view keyword : keywords)
        {
            if ((keyword == "dependencies") != (m_dialect == JsonSchemaDialect::Draft7))
            {
                continue;
            }
            const auto* dependencies = schema.if_contains(keyword);
            if (dependencies == nullptr)
            {
                continue;
            }
            const auto keywordPath = ChildPath(schemaPath, keyword);
            for (const auto& dependency : dependencies->as_object())
            {
                if (!instance.contains(dependency.key()))
                {
                    continue;
                }
                const auto dependencyPath = ChildPath(keywordPath, dependency.key());
                if (keyword == "dependentRequired" ||
                    (keyword == "dependencies" && dependency.value().is_array()))
                {
                    for (const auto& required : dependency.value().as_array())
                    {
                        const auto requiredName = ToStringView(required.as_string());
                        if (!instance.contains(requiredName))
                        {
                            AddError(ChildPath(instancePath, requiredName), dependencyPath,
                                     "dependent property is missing");
                        }
                    }
                }
                else
                {
                    Validate(dependency.value(), instanceValue, instancePath,
                             dependencyPath, depth + 1);
                }
            }
        }
    }

    [[nodiscard]] std::size_t
    ValidateTupleItems(const JsonArray& schemas, const JsonArray& instance,
                       std::string_view instancePath, std::string_view schemaPath,
                       std::size_t depth)
    {
        const auto count = std::min(instance.size(), schemas.size());
        for (std::size_t index = 0; index < count; ++index)
        {
            const auto token = std::to_string(index);
            Validate(schemas[index], instance[index], ChildPath(instancePath, token),
                     ChildPath(schemaPath, token), depth + 1);
        }
        return count;
    }

    void ValidateArray(const JsonObject& schema, const JsonArray& instance,
                       std::string_view instancePath, std::string_view schemaPath,
                       std::size_t depth)
    {
        ValidateSizeKeyword(schema, "minItems", instance.size(), true, instancePath,
                            schemaPath);
        ValidateSizeKeyword(schema, "maxItems", instance.size(), false, instancePath,
                            schemaPath);

        std::size_t itemStart = 0;
        if (m_dialect == JsonSchemaDialect::Draft2020_12)
        {
            if (const auto* prefixItems = schema.if_contains("prefixItems");
                prefixItems != nullptr && prefixItems->is_array())
            {
                itemStart = ValidateTupleItems(prefixItems->as_array(), instance, instancePath,
                                              ChildPath(schemaPath, "prefixItems"), depth);
            }
        }

        if (const auto* unique = schema.if_contains("uniqueItems"))
        {
            const auto keywordPath = ChildPath(schemaPath, "uniqueItems");
            if (!unique->is_bool())
            {
                AddError(instancePath, keywordPath, "uniqueItems must be a boolean");
            }
            else if (unique->as_bool())
            {
                bool duplicateFound = false;
                for (std::size_t first = 0; first < instance.size(); ++first)
                {
                    for (std::size_t second = first + 1; second < instance.size(); ++second)
                    {
                        if (JsonSchemaEqual(instance[first], instance[second]))
                        {
                            AddError(ChildPath(instancePath, std::to_string(second)),
                                     keywordPath, "array items must be unique");
                            duplicateFound = true;
                            break;
                        }
                    }
                    if (duplicateFound)
                    {
                        break;
                    }
                }
            }
        }

        if (const auto* items = schema.if_contains("items"))
        {
            const auto keywordPath = ChildPath(schemaPath, "items");
            if (items->is_array())
            {
                itemStart = ValidateTupleItems(items->as_array(), instance, instancePath,
                                              keywordPath, depth);
                if (const auto* additional = schema.if_contains("additionalItems"))
                {
                    const auto additionalPath = ChildPath(schemaPath, "additionalItems");
                    for (std::size_t index = itemStart; index < instance.size(); ++index)
                    {
                        Validate(*additional, instance[index],
                                 ChildPath(instancePath, std::to_string(index)),
                                 additionalPath, depth + 1);
                    }
                }
            }
            else
            {
                for (std::size_t index = itemStart; index < instance.size(); ++index)
                {
                    Validate(*items, instance[index],
                             ChildPath(instancePath, std::to_string(index)), keywordPath,
                             depth + 1);
                }
            }
        }

        ValidateContains(schema, instance, instancePath, schemaPath, depth);
    }

    void ValidateContains(const JsonObject& schema, const JsonArray& instance,
                          std::string_view instancePath, std::string_view schemaPath,
                          std::size_t depth)
    {
        const auto* contains = schema.if_contains("contains");
        if (contains == nullptr)
        {
            return;
        }

        std::size_t matches = 0;
        const auto keywordPath = ChildPath(schemaPath, "contains");
        for (std::size_t index = 0; index < instance.size(); ++index)
        {
            const auto itemMatches =
                BranchMatches(*contains, instance[index],
                              ChildPath(instancePath, std::to_string(index)), keywordPath,
                              depth + 1);
            if (!itemMatches)
            {
                return;
            }
            if (*itemMatches)
            {
                ++matches;
            }
        }

        std::size_t minimum = 1;
        if (const auto* minContains = schema.if_contains("minContains"))
        {
            minimum = *NonNegativeSize(*minContains);
        }
        std::size_t maximum = std::numeric_limits<std::size_t>::max();
        if (const auto* maxContains = schema.if_contains("maxContains"))
        {
            maximum = *NonNegativeSize(*maxContains);
        }
        if (matches < minimum)
        {
            AddError(instancePath,
                     schema.contains("minContains")
                         ? ChildPath(schemaPath, "minContains")
                         : keywordPath,
                     "array contains fewer matching items than required");
        }
        else if (matches > maximum)
        {
            AddError(instancePath, ChildPath(schemaPath, "maxContains"),
                     "array contains more matching items than allowed");
        }
    }

    void ValidateString(const JsonObject& schema, const JsonString& instance,
                        std::string_view instancePath, std::string_view schemaPath)
    {
        const auto value = ToStringView(instance);
        const auto length = Utf8CodePointCount(value);
        ValidateSizeKeyword(schema, "minLength", length, true, instancePath, schemaPath);
        ValidateSizeKeyword(schema, "maxLength", length, false, instancePath, schemaPath);

        if (const auto* pattern = schema.if_contains("pattern"))
        {
            const auto keywordPath = ChildPath(schemaPath, "pattern");
            if (!pattern->is_string())
            {
                AddError(instancePath, keywordPath, "pattern must be a string");
                return;
            }
            const auto expression = CompilePattern(ToStringView(pattern->as_string()),
                                                   instancePath, keywordPath);
            if (expression)
            {
                if (!MatchesPattern(*expression, value, instancePath, keywordPath))
                {
                    AddError(instancePath, keywordPath,
                             "string does not match the required pattern");
                }
            }
        }
    }

    void ValidateNumber(const JsonObject& schema, const JsonValue& instance,
                        std::string_view instancePath, std::string_view schemaPath)
    {
        if (instance.is_double() && !std::isfinite(instance.as_double()))
        {
            AddError(instancePath, schemaPath, "number must be finite");
            return;
        }

        ValidateNumberLimit(schema, "minimum", instance, instancePath, schemaPath, false,
                            false);
        ValidateNumberLimit(schema, "maximum", instance, instancePath, schemaPath, true,
                            false);
        ValidateNumberLimit(schema, "exclusiveMinimum", instance, instancePath, schemaPath,
                            false, true);
        ValidateNumberLimit(schema, "exclusiveMaximum", instance, instancePath, schemaPath,
                            true, true);

        if (const auto* multipleOf = schema.if_contains("multipleOf"))
        {
            const auto keywordPath = ChildPath(schemaPath, "multipleOf");
            if (!IsNumber(*multipleOf) || AsNumber(*multipleOf) <= 0)
            {
                AddError(instancePath, keywordPath, "multipleOf must be a positive number");
                return;
            }
            bool isMultiple = false;
            if (!instance.is_double())
            {
                const auto value = IntegerMagnitude(instance);
                if (!multipleOf->is_double())
                {
                    isMultiple = value % IntegerMagnitude(*multipleOf) == 0;
                }
                else
                {
                    constexpr double uint64Limit = 18446744073709551616.0;
                    const double divisor = multipleOf->as_double();
                    isMultiple = divisor >= uint64Limit
                                     ? value == 0
                                     : value % static_cast<std::uint64_t>(divisor) == 0;
                }
            }
            else
            {
                const auto divisor = AsNumber(*multipleOf);
                const auto remainder = std::fmod(std::fabs(AsNumber(instance)), divisor);
                isMultiple = remainder == 0;
            }
            if (!isMultiple)
            {
                AddError(instancePath, keywordPath,
                         "number is not a multiple of the required value");
            }
        }
    }

    void ValidateSizeKeyword(const JsonObject& schema, std::string_view keyword,
                             std::size_t actual, bool minimum, std::string_view instancePath,
                             std::string_view schemaPath)
    {
        const auto* constraint = schema.if_contains(keyword);
        if (constraint == nullptr)
        {
            return;
        }
        const auto keywordPath = ChildPath(schemaPath, keyword);
        const auto expected = NonNegativeSize(*constraint);
        if (!expected)
        {
            AddError(instancePath, keywordPath,
                     std::string(keyword) + " must be a non-negative integer");
        }
        else if ((minimum && actual < *expected) || (!minimum && actual > *expected))
        {
            AddError(instancePath, keywordPath,
                     minimum ? "value has fewer elements than allowed"
                             : "value has more elements than allowed");
        }
    }

    void ValidateNumberLimit(const JsonObject& schema, std::string_view keyword,
                             const JsonValue& instance, std::string_view instancePath,
                             std::string_view schemaPath, bool maximum, bool exclusive)
    {
        const auto* limit = schema.if_contains(keyword);
        if (limit == nullptr)
        {
            return;
        }
        const auto keywordPath = ChildPath(schemaPath, keyword);
        if (!IsNumber(*limit))
        {
            AddError(instancePath, keywordPath, std::string(keyword) + " must be a number");
            return;
        }
        const auto comparison = CompareNumbers(instance, *limit);
        const bool fails = maximum ? (exclusive ? comparison >= 0 : comparison > 0)
                                   : (exclusive ? comparison <= 0 : comparison < 0);
        if (fails)
        {
            AddError(instancePath, keywordPath,
                     maximum ? "number is greater than the allowed maximum"
                             : "number is less than the allowed minimum");
        }
    }

    JsonSchemaDialect m_dialect;
    JsonSchemaValidationOptions m_options;
    const JsonValue& m_rootSchema;
    JsonSchemaValidationResult m_result;
    std::optional<JsonSchemaCompileError> m_compileError;
    std::vector<const JsonValue*> m_checkedSchemas;
    bool m_checkingSchema = false;
    bool m_resourceError = false;
};

} // namespace

Result<JsonValue, JsonErrorCode> ParseJson(std::string_view text)
{
    JsonParseOptions options = {};
    return ParseJson(text, options);
}

Result<JsonValue, JsonErrorCode> ParseJson(std::string_view text,
                                          const JsonParseOptions& options)
{
    JsonErrorCode error;
    auto value = boost::json::parse({text.data(), text.size()}, error, {}, options);
    if (error)
    {
        return Failure(error);
    }
    return Success(std::move(value));
}

std::string PrettyJson(const JsonValue& value, std::string_view indent)
{
    std::string output;
    std::string currentIndent;
    PrettyJsonImpl(value, indent, output, currentIndent);
    return output;
}

JsonSchema::JsonSchema(JsonValue schema, JsonSchemaDialect dialect)
    : m_schema(std::move(schema)), m_dialect(dialect)
{
}

Result<JsonSchema, JsonSchemaCompileError>
JsonSchema::CompileFile(const FilePath& path, JsonSchemaDialect dialect)
{
    const auto text = File::ReadAllText(path);
    if (!text)
    {
        return Failure(JsonSchemaCompileError{
            JsonSchemaCompileErrorCode::FileReadError,
            dialect,
            {},
            "unable to read schema file: " + path.string(),
        });
    }

    const auto schema = ParseJson(*text);
    if (!schema)
    {
        return Failure(JsonSchemaCompileError{
            JsonSchemaCompileErrorCode::InvalidJson,
            dialect,
            {},
            "unable to parse schema file: " + schema.error().message(),
        });
    }

    return Compile(schema.value(), dialect);
}

Result<JsonSchema, JsonSchemaCompileError>
JsonSchema::Compile(const JsonValue& schema)
{
    if (!schema.is_object() || !schema.as_object().contains("$schema"))
    {
        return Failure(JsonSchemaCompileError{
            JsonSchemaCompileErrorCode::MissingDialect,
            std::nullopt,
            {},
            "schema does not declare $schema",
        });
    }

    const auto& declaredValue = schema.as_object().at("$schema");
    if (!declaredValue.is_string())
    {
        return Failure(JsonSchemaCompileError{
            JsonSchemaCompileErrorCode::InvalidSchema,
            std::nullopt,
            "/$schema",
            "$schema must be a string",
        });
    }

    auto declared = ToStringView(declaredValue.as_string());
    if (declared.ends_with('#'))
    {
        declared.remove_suffix(1);
    }

    if (declared == "http://json-schema.org/draft-07/schema")
    {
        return Compile(schema, JsonSchemaDialect::Draft7);
    }
    if (declared == "https://json-schema.org/draft/2019-09/schema")
    {
        return Compile(schema, JsonSchemaDialect::Draft2019_09);
    }
    if (declared == "https://json-schema.org/draft/2020-12/schema")
    {
        return Compile(schema, JsonSchemaDialect::Draft2020_12);
    }

    return Failure(JsonSchemaCompileError{
        JsonSchemaCompileErrorCode::UnsupportedDialect,
        std::nullopt,
        "/$schema",
        "schema dialect is not supported: " + std::string(declared),
    });
}

Result<JsonSchema, JsonSchemaCompileError>
JsonSchema::Compile(const JsonValue& schema, JsonSchemaDialect dialect)
{
    if (dialect != JsonSchemaDialect::Draft7 &&
        dialect != JsonSchemaDialect::Draft2019_09 &&
        dialect != JsonSchemaDialect::Draft2020_12)
    {
        return Failure(JsonSchemaCompileError{
            JsonSchemaCompileErrorCode::UnsupportedDialect,
            dialect,
            {},
            "schema dialect is not supported",
        });
    }

    const JsonSchemaValidationOptions options;
    JsonSchemaValidator validator(dialect, options, schema);
    auto error = validator.CheckSchema(schema);
    if (error)
    {
        return Failure(std::move(*error));
    }

    return Success(JsonSchema(schema, dialect));
}

JsonSchemaDialect JsonSchema::Dialect() const noexcept
{
    return m_dialect;
}

JsonSchemaValidationResult
JsonSchema::Validate(const JsonValue& instance,
                     const JsonSchemaValidationOptions& options) const
{
    JsonSchemaValidator validator(m_dialect, options, m_schema);
    return validator.ValidateInstance(m_schema, instance);
}

} // namespace rad
