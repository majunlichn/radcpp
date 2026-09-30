#include <rad/IO/File.h>
#include <rad/IO/Json.h>

#include "JsonSchemaRegex.h"
#include "JsonSchemaReferences.h"
#include "JsonSchemaMetaSchemas.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <iterator>
#include <optional>
#include <numeric>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace rad
{
namespace
{

[[nodiscard]] std::string_view ToStringView(const JsonString& value) noexcept
{
    return {value.data(), value.size()};
}

using detail::ChildPath;

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

struct DecimalNumber
{
    std::uint64_t coefficient;
    int exponent;
};

[[nodiscard]] Result<DecimalNumber, std::string> AsDecimal(const JsonValue& value)
{
    if (!value.is_double())
    {
        auto coefficient = IntegerMagnitude(value);
        int exponent = 0;
        while (coefficient != 0 && coefficient % 10 == 0)
        {
            coefficient /= 10;
            ++exponent;
        }
        return Success(DecimalNumber{coefficient, exponent});
    }

    std::array<char, 64> buffer;
    const auto [end, error] =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), std::fabs(value.as_double()),
                      std::chars_format::general);
    if (error != std::errc{})
    {
        return Failure(std::string("unable to convert number to a round-trip decimal"));
    }
    const std::string_view text(buffer.data(), static_cast<std::size_t>(end - buffer.data()));
    const auto exponentStart = text.find_first_of("eE");
    const auto mantissa = text.substr(0, exponentStart);
    int exponent = 0;
    if (exponentStart != std::string_view::npos)
    {
        auto exponentText = text.substr(exponentStart + 1);
        if (exponentText.starts_with('+'))
        {
            exponentText.remove_prefix(1);
        }
        const auto [next, exponentError] = std::from_chars(
            exponentText.data(), exponentText.data() + exponentText.size(), exponent);
        if (exponentError != std::errc{} || next != exponentText.data() + exponentText.size())
        {
            return Failure(std::string("unable to decode round-trip decimal exponent"));
        }
    }
    const auto point = mantissa.find('.');
    if (point != std::string_view::npos)
    {
        exponent -= static_cast<int>(mantissa.size() - point - 1);
    }
    std::array<char, 64> digits;
    std::size_t count = 0;
    for (const char c : mantissa)
    {
        if (c != '.')
        {
            digits[count++] = c;
        }
    }
    while (count != 0 && digits[count - 1] == '0')
    {
        --count;
        ++exponent;
    }
    if (count == 0)
    {
        return Success(DecimalNumber{0, 0});
    }
    std::uint64_t coefficient = 0;
    const auto [next, coefficientError] =
        std::from_chars(digits.data(), digits.data() + count, coefficient);
    if (coefficientError != std::errc{} || next != digits.data() + count)
    {
        return Failure(std::string("unable to decode round-trip decimal coefficient"));
    }
    return Success(DecimalNumber{coefficient, exponent});
}

[[nodiscard]] bool IsDecimalMultiple(DecimalNumber value, DecimalNumber divisor) noexcept
{
    if (value.coefficient == 0)
    {
        return true;
    }
    const auto common = std::gcd(value.coefficient, divisor.coefficient);
    auto numerator = value.coefficient / common;
    auto denominator = divisor.coefficient / common;
    int power = value.exponent - divisor.exponent;
    if (power < 0)
    {
        if (denominator != 1)
        {
            return false;
        }
        while (power < 0)
        {
            if (numerator % 10 != 0)
            {
                return false;
            }
            numerator /= 10;
            ++power;
        }
        return true;
    }

    // The reduced denominator must divide 10^power, whose only prime factors are 2 and 5.
    int twos = 0;
    int fives = 0;
    while (denominator % 2 == 0)
    {
        denominator /= 2;
        ++twos;
    }
    while (denominator % 5 == 0)
    {
        denominator /= 5;
        ++fives;
    }
    return denominator == 1 && twos <= power && fives <= power;
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

class JsonSchemaKeywords
{
public:
    JsonSchemaKeywords(const JsonObject& object,
                       const detail::JsonSchemaVocabularyProfile& profile) :
        m_object(object),
        m_profile(profile)
    {
    }

    [[nodiscard]] const JsonValue* if_contains(std::string_view keyword) const
    {
        return m_profile.IsKeywordEnabled(keyword) ? m_object.if_contains(keyword) : nullptr;
    }

    [[nodiscard]] bool contains(std::string_view keyword) const
    {
        return if_contains(keyword) != nullptr;
    }

private:
    const JsonObject& m_object;
    const detail::JsonSchemaVocabularyProfile& m_profile;
}; // class JsonSchemaKeywords

class JsonSchemaValidator
{
public:
    JsonSchemaValidator(JsonSchemaDialect dialect,
                        const JsonSchemaValidationOptions& options,
                        const JsonValue& rootSchema,
                        const detail::JsonSchemaReferences& references)
        : m_dialect(dialect), m_options(options), m_rootSchema(rootSchema),
          m_references(references)
    {
    }

    [[nodiscard]] Result<detail::JsonSchemaReferences::Patterns, JsonSchemaCompileError>
    CheckSchema(const JsonValue& schema)
    {
        m_checkingSchema = true;
        ValidateSchemaDefinition(schema, "/0", 0);
        if (m_compileError)
        {
            return Failure(std::move(*m_compileError));
        }
        return Success(std::move(m_patterns));
    }

    [[nodiscard]] JsonSchemaValidationResult ValidateInstance(const JsonValue& schema,
                                                              const JsonValue& instance)
    {
        Validate(schema, instance, {}, "/0", 0);
        return std::move(m_result);
    }

private:
    struct Evaluation
    {
        bool valid = true;
        std::vector<bool> properties;
        std::vector<bool> items;

        static void Mark(std::vector<bool>& locations, std::size_t index)
        {
            if (locations.size() <= index)
            {
                locations.resize(index + 1, false);
            }
            locations[index] = true;
        }

        void Merge(const Evaluation& other)
        {
            if (!other.valid)
            {
                return;
            }
            const auto merge = [](std::vector<bool>& target, const std::vector<bool>& source) {
                if (target.size() < source.size())
                {
                    target.resize(source.size(), false);
                }
                for (std::size_t index = 0; index < source.size(); ++index)
                {
                    target[index] = target[index] || source[index];
                }
            };
            merge(properties, other.properties);
            merge(items, other.items);
        }
    };

    struct ReferenceTarget
    {
        const JsonValue* schema;
        std::string path;
    };

    struct ResourceScope
    {
        std::vector<const std::string*>& scope;
        bool pushed;

        ResourceScope(std::vector<const std::string*>& scope, const std::string* resource) :
            scope(scope),
            pushed(resource != nullptr && (scope.empty() || *scope.back() != *resource))
        {
            if (pushed)
            {
                scope.push_back(resource);
            }
        }

        ~ResourceScope()
        {
            if (pushed)
            {
                scope.pop_back();
            }
        }
    };

    [[nodiscard]] std::optional<ReferenceTarget>
    ResolveReference(const JsonValue& reference, std::string_view schemaPath,
                     std::string_view keyword = "$ref")
    {
        if (!reference.is_string())
        {
            AddError({}, schemaPath, std::string(keyword) + " must be a string");
            return std::nullopt;
        }
        const auto* resolution = m_references.Find(schemaPath);
        if (resolution != nullptr && !*resolution)
        {
            const auto& error = resolution->error();
            if (m_checkingSchema)
            {
                if (!m_compileError)
                {
                    m_compileError = error;
                }
            }
            else
            {
                ++m_errorCount;
                if (m_result.errors.size() < std::max<std::size_t>(m_options.maxErrors, 1))
                {
                    m_result.errors.push_back({{}, error.schemaPath, error.message, error.schemaUri});
                }
            }
            return std::nullopt;
        }
        const auto* targetPath = resolution ? &resolution->value().path : nullptr;
        if (resolution != nullptr && !m_checkingSchema && keyword == "$dynamicRef" &&
            !resolution->value().dynamicAnchor.empty())
        {
            for (const auto* resource : m_dynamicScope)
            {
                const auto* anchors = m_references.DynamicAnchors(*resource);
                if (anchors == nullptr)
                {
                    continue;
                }
                const auto anchor = anchors->find(resolution->value().dynamicAnchor);
                if (anchor != anchors->end())
                {
                    targetPath = &anchor->second;
                    break;
                }
            }
        }
        const auto* target =
            targetPath ? detail::FindJsonSchemaValue(m_rootSchema, *targetPath) : nullptr;
        if (target == nullptr)
        {
            AddError({}, schemaPath, "compiled " + std::string(keyword) + " target does not exist");
            return std::nullopt;
        }
        return ReferenceTarget{target, *targetPath};
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
                    m_references.SchemaPath(schemaPath),
                    std::move(message),
                    m_references.SchemaUri(schemaPath),
                };
            }
            return;
        }
        ++m_errorCount;
        if (m_result.errors.size() >= std::max<std::size_t>(m_options.maxErrors, 1))
        {
            return;
        }
        m_result.errors.push_back(
            {std::string(instancePath), m_references.SchemaPath(schemaPath), std::move(message),
             m_references.SchemaUri(schemaPath)});
    }

    void AddUnsupportedError(std::string_view instancePath, std::string_view schemaPath,
                             std::string message)
    {
        if (m_checkingSchema && !m_compileError)
        {
            m_compileError = JsonSchemaCompileError{
                JsonSchemaCompileErrorCode::UnsupportedFeature,
                m_dialect,
                m_references.SchemaPath(schemaPath),
                std::move(message),
                m_references.SchemaUri(schemaPath),
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
        if (!m_resourceDiagnostic)
        {
            m_resourceDiagnostic = JsonSchemaValidationError{
                std::string(instancePath), m_references.SchemaPath(schemaPath), message,
                m_references.SchemaUri(schemaPath)};
        }
        AddError(instancePath, schemaPath, std::move(message));
    }

    void AddRegexError(const detail::JsonSchemaRegexError& error,
                       std::string_view instancePath, std::string_view schemaPath)
    {
        if (error.resource)
        {
            AddResourceError(instancePath, schemaPath, error.message);
        }
        else if (error.unsupported)
        {
            AddUnsupportedError(instancePath, schemaPath, error.message);
        }
        else
        {
            AddError(instancePath, schemaPath, error.message);
        }
    }

    void CheckPattern(std::string_view pattern, std::string_view schemaPath)
    {
        if (m_patterns.contains(std::string(schemaPath)))
        {
            return;
        }
        auto expression = detail::JsonSchemaRegex::Compile(pattern);
        if (!expression)
        {
            AddRegexError(expression.error(), {}, schemaPath);
            return;
        }
        m_patterns.emplace(std::string(schemaPath), std::move(expression.value()));
    }

    [[nodiscard]] const detail::JsonSchemaRegex*
    FindPattern(std::string_view instancePath, std::string_view schemaPath)
    {
        const auto* expression = m_references.FindPattern(schemaPath);
        if (expression == nullptr)
        {
            AddResourceError(instancePath, schemaPath, "compiled regular expression does not exist");
        }
        return expression;
    }

    [[nodiscard]] bool MatchesPattern(const detail::JsonSchemaRegex& expression,
                                     std::string_view value, std::string_view instancePath,
                                     std::string_view schemaPath)
    {
        const auto result = expression.Matches(value);
        if (!result)
        {
            AddRegexError(result.error(), instancePath, schemaPath);
            return false;
        }
        return result.value();
    }

    void ValidateSchemaDefinition(const JsonValue& schema, std::string_view schemaPath,
                                  std::size_t depth)
    {
        if (m_compileError)
        {
            return;
        }
        if (const auto* error = m_references.ProfileError(schemaPath))
        {
            m_compileError = *error;
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

        if (m_dialect == JsonSchemaDialect::Draft2020_12)
        {
            const auto* resource = m_references.Resource(schemaPath);
            if (resource == nullptr)
            {
                AddError({}, schemaPath, "compiled schema resource does not exist");
                return;
            }
            if (std::find(m_checkedResources.begin(), m_checkedResources.end(), *resource) ==
                m_checkedResources.end())
            {
                m_checkedResources.push_back(*resource);
                // Dynamic references can reach anchors outside the static validation path.
                if (const auto* anchors = m_references.DynamicAnchors(*resource))
                {
                    for (const auto& [name, path] : *anchors)
                    {
                        const auto* target = detail::FindJsonSchemaValue(m_rootSchema, path);
                        if (target == nullptr)
                        {
                            AddError({}, path, "compiled dynamic anchor does not exist");
                            return;
                        }
                        ValidateSchemaDefinition(*target, path, depth + 1);
                    }
                }
            }
        }
        const auto* profile = m_references.Profile(schemaPath);
        if (profile == nullptr)
        {
            AddError({}, schemaPath, "compiled schema vocabulary profile does not exist");
            return;
        }
        const JsonSchemaKeywords object(schema.as_object(), *profile);
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
        if (m_dialect == JsonSchemaDialect::Draft2019_09)
        {
            if (const auto* anchor = object.if_contains("$recursiveAnchor");
                anchor != nullptr && !anchor->is_bool())
            {
                AddError({}, ChildPath(schemaPath, "$recursiveAnchor"),
                         "$recursiveAnchor must be a boolean");
            }
            if (const auto* reference = object.if_contains("$recursiveRef"))
            {
                const auto target = ResolveReference(
                    *reference, ChildPath(schemaPath, "$recursiveRef"), "$recursiveRef");
                if (!target)
                {
                    return;
                }
                ValidateSchemaDefinition(*target->schema, target->path, depth + 1);
            }
        }
        else if (m_dialect == JsonSchemaDialect::Draft2020_12)
        {
            if (const auto* reference = object.if_contains("$dynamicRef"))
            {
                const auto target = ResolveReference(
                    *reference, ChildPath(schemaPath, "$dynamicRef"), "$dynamicRef");
                if (!target)
                {
                    return;
                }
                ValidateSchemaDefinition(*target->schema, target->path, depth + 1);
            }
        }
        ValidateVocabulary(object, {}, schemaPath);

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
                CheckPattern(ToStringView(pattern->as_string()), patternPath);
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
                    CheckPattern(pattern.key(), patternPath);
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
        if (m_dialect != JsonSchemaDialect::Draft7)
        {
            constexpr std::array unevaluatedKeywords = {
                "unevaluatedProperties", "unevaluatedItems",
            };
            for (const std::string_view keyword : unevaluatedKeywords)
            {
                if (const auto* unevaluated = object.if_contains(keyword))
                {
                    ValidateSchemaDefinition(*unevaluated, ChildPath(schemaPath, keyword),
                                             depth + 1);
                }
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

    void ValidateDependenciesSchema(const JsonSchemaKeywords& schema,
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

    [[nodiscard]] std::optional<Evaluation>
    EvaluateBranch(const JsonValue& schema, const JsonValue& instance,
                   std::string_view instancePath, std::string_view schemaPath,
                   std::size_t depth)
    {
        JsonSchemaValidationOptions options = m_options;
        options.maxErrors = 1;
        JsonSchemaValidator validator(m_dialect, options, m_rootSchema, m_references);
        validator.m_dynamicScope = m_dynamicScope;
        auto evaluation = validator.Validate(schema, instance, instancePath, schemaPath, depth);
        if (validator.m_resourceError)
        {
            const auto& error = *validator.m_resourceDiagnostic;
            m_resourceError = true;
            if (!m_resourceDiagnostic)
            {
                m_resourceDiagnostic = error;
            }
            ++m_errorCount;
            if (m_result.errors.size() < std::max<std::size_t>(m_options.maxErrors, 1))
            {
                m_result.errors.push_back(error);
            }
            return std::nullopt;
        }
        return evaluation;
    }

    Evaluation Validate(const JsonValue& schema, const JsonValue& instance,
                        std::string_view instancePath, std::string_view schemaPath,
                        std::size_t depth)
    {
        Evaluation evaluation;
        const auto errorsBefore = m_errorCount;
        if (m_resourceError)
        {
            evaluation.valid = false;
            return evaluation;
        }
        if (depth > m_options.maxDepth)
        {
            AddResourceError(instancePath, schemaPath,
                             "maximum validation depth exceeded");
            evaluation.valid = false;
            return evaluation;
        }
        if (schema.is_bool())
        {
            if (!schema.as_bool())
            {
                AddError(instancePath, schemaPath, "value is rejected by the false schema");
            }
            evaluation.valid = schema.as_bool();
            return evaluation;
        }
        if (!schema.is_object())
        {
            AddError(instancePath, schemaPath, "schema must be an object or boolean");
            evaluation.valid = false;
            return evaluation;
        }

        const auto* resource = m_dialect != JsonSchemaDialect::Draft7
                                   ? m_references.Resource(schemaPath)
                                   : nullptr;
        if (m_dialect != JsonSchemaDialect::Draft7 && resource == nullptr)
        {
            AddResourceError(instancePath, schemaPath, "compiled schema resource does not exist");
            evaluation.valid = false;
            return evaluation;
        }
        ResourceScope resourceScope(m_dynamicScope, resource);
        const auto* profile = m_references.Profile(schemaPath);
        if (profile == nullptr)
        {
            AddResourceError(instancePath, schemaPath,
                             "compiled schema vocabulary profile does not exist");
            evaluation.valid = false;
            return evaluation;
        }
        const JsonSchemaKeywords object(schema.as_object(), *profile);
        if (const auto* reference = object.if_contains("$ref"))
        {
            const auto target = ResolveReference(*reference, ChildPath(schemaPath, "$ref"));
            if (!target)
            {
                evaluation.valid = false;
                return evaluation;
            }
            const auto referenced = Validate(*target->schema, instance, instancePath,
                                              target->path, depth + 1);
            evaluation.Merge(referenced);
            if (m_dialect == JsonSchemaDialect::Draft7)
            {
                return referenced;
            }
        }
        if (m_dialect == JsonSchemaDialect::Draft2019_09)
        {
            if (const auto* reference = object.if_contains("$recursiveRef"))
            {
                auto target = ResolveReference(
                    *reference, ChildPath(schemaPath, "$recursiveRef"), "$recursiveRef");
                if (!target)
                {
                    evaluation.valid = false;
                    return evaluation;
                }
                if (m_references.HasRecursiveAnchor(target->path))
                {
                    // An unanchored resource ends the chain of recursive extensions.
                    for (auto scope = m_dynamicScope.rbegin(); scope != m_dynamicScope.rend();
                         ++scope)
                    {
                        if (!m_references.HasRecursiveAnchor(**scope))
                        {
                            break;
                        }
                        target->path = **scope;
                    }
                    target->schema = detail::FindJsonSchemaValue(m_rootSchema, target->path);
                    if (target->schema == nullptr)
                    {
                        AddResourceError(instancePath, ChildPath(schemaPath, "$recursiveRef"),
                                         "compiled recursive resource does not exist");
                        evaluation.valid = false;
                        return evaluation;
                    }
                }
                evaluation.Merge(Validate(*target->schema, instance, instancePath,
                                          target->path, depth + 1));
            }
        }
        else if (m_dialect == JsonSchemaDialect::Draft2020_12)
        {
            if (const auto* reference = object.if_contains("$dynamicRef"))
            {
                const auto target = ResolveReference(
                    *reference, ChildPath(schemaPath, "$dynamicRef"), "$dynamicRef");
                if (!target)
                {
                    evaluation.valid = false;
                    return evaluation;
                }
                evaluation.Merge(Validate(*target->schema, instance, instancePath,
                                          target->path, depth + 1));
            }
        }
        ValidateVocabulary(object, instancePath, schemaPath);
        ValidateType(object, instance, instancePath, schemaPath);
        ValidateEnumAndConst(object, instance, instancePath, schemaPath);
        ValidateCompositions(object, instance, instancePath, schemaPath, depth, evaluation);

        if (instance.is_object())
        {
            ValidateObject(object, instance, instancePath, schemaPath, depth, evaluation);
        }
        if (instance.is_array())
        {
            ValidateArray(object, instance.as_array(), instancePath, schemaPath, depth, evaluation);
        }
        if (instance.is_string())
        {
            ValidateString(object, instance.as_string(), instancePath, schemaPath);
        }
        if (IsNumber(instance))
        {
            ValidateNumber(object, instance, instancePath, schemaPath);
        }
        evaluation.valid = m_errorCount == errorsBefore && !m_resourceError;
        return evaluation;
    }

    void ValidateVocabulary(const JsonSchemaKeywords& schema, std::string_view instancePath,
                            std::string_view schemaPath)
    {
        if (m_dialect == JsonSchemaDialect::Draft7)
        {
            return;
        }
        const auto* vocabulary = schema.if_contains("$vocabulary");
        if (vocabulary == nullptr)
        {
            return;
        }
        const auto keywordPath = ChildPath(schemaPath, "$vocabulary");
        if (!vocabulary->is_object())
        {
            AddError(instancePath, keywordPath, "$vocabulary must be an object");
            return;
        }
        const auto coreUri = detail::JsonSchemaCoreVocabularyUri(m_dialect);
        const auto* core = vocabulary->as_object().if_contains(coreUri);
        if (core == nullptr)
        {
            AddError(instancePath, keywordPath, "$vocabulary must require the core vocabulary");
            return;
        }
        if (!core->is_bool() || !core->as_bool())
        {
            AddError(instancePath, ChildPath(keywordPath, coreUri),
                     "core vocabulary must be required (true)");
            return;
        }
        for (const auto& entry : vocabulary->as_object())
        {
            const auto entryPath = ChildPath(keywordPath, entry.key());
            if (!entry.value().is_bool())
            {
                AddError(instancePath, entryPath, "vocabulary requirement must be a boolean");
                continue;
            }
            const auto supported = detail::GetJsonSchemaVocabularySupport(entry.key(), m_dialect);
            if (!supported)
            {
                AddError(instancePath, entryPath, supported.error());
            }
            else if (entry.value().as_bool() && !supported.value())
            {
                AddUnsupportedError(instancePath, entryPath,
                                    "required vocabulary is not supported: " +
                                        std::string(entry.key()));
            }
        }
    }

    void ValidateType(const JsonSchemaKeywords& schema, const JsonValue& instance,
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

    void ValidateEnumAndConst(const JsonSchemaKeywords& schema, const JsonValue& instance,
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

    void ValidateCompositions(const JsonSchemaKeywords& schema, const JsonValue& instance,
                              std::string_view instancePath, std::string_view schemaPath,
                              std::size_t depth, Evaluation& evaluation)
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
                Evaluation combined;
                for (std::size_t index = 0; index < allOf->as_array().size(); ++index)
                {
                    const auto branch = Validate(allOf->as_array()[index], instance, instancePath,
                                                 ChildPath(keywordPath, std::to_string(index)),
                                                 depth + 1);
                    combined.Merge(branch);
                    combined.valid = combined.valid && branch.valid;
                }
                evaluation.Merge(combined);
            }
        }

        ValidateAlternative(schema, "anyOf", instance, instancePath, schemaPath, depth, false,
                            evaluation);
        ValidateAlternative(schema, "oneOf", instance, instancePath, schemaPath, depth, true,
                            evaluation);

        if (const auto* notSchema = schema.if_contains("not"))
        {
            const auto keywordPath = ChildPath(schemaPath, "not");
            const auto matches =
                EvaluateBranch(*notSchema, instance, instancePath, keywordPath, depth + 1);
            if (!matches)
            {
                return;
            }
            if (matches->valid)
            {
                AddError(instancePath, keywordPath, "value matches the disallowed schema");
            }
        }

        if (const auto* condition = schema.if_contains("if"))
        {
            const auto matches =
                EvaluateBranch(*condition, instance, instancePath,
                              ChildPath(schemaPath, "if"), depth + 1);
            if (!matches)
            {
                return;
            }
            evaluation.Merge(*matches);
            const std::string_view keyword = matches->valid ? "then" : "else";
            if (const auto* branch = schema.if_contains(keyword))
            {
                evaluation.Merge(Validate(*branch, instance, instancePath,
                                          ChildPath(schemaPath, keyword), depth + 1));
            }
        }
    }

    void ValidateAlternative(const JsonSchemaKeywords& schema, std::string_view keyword,
                             const JsonValue& instance, std::string_view instancePath,
                             std::string_view schemaPath, std::size_t depth, bool exactlyOne,
                             Evaluation& evaluation)
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
        Evaluation combined;
        for (std::size_t index = 0; index < alternatives->as_array().size(); ++index)
        {
            const auto branchMatches =
                EvaluateBranch(alternatives->as_array()[index], instance, instancePath,
                              ChildPath(keywordPath, std::to_string(index)), depth + 1);
            if (!branchMatches)
            {
                return;
            }
            if (branchMatches->valid)
            {
                ++matches;
                combined.Merge(*branchMatches);
            }
        }
        if ((!exactlyOne && matches == 0) || (exactlyOne && matches != 1))
        {
            AddError(instancePath, keywordPath,
                     exactlyOne ? "value must match exactly one schema"
                                : "value must match at least one schema");
        }
        else
        {
            evaluation.Merge(combined);
        }
    }

    void ValidateObject(const JsonSchemaKeywords& schema, const JsonValue& instanceValue,
                        std::string_view instancePath, std::string_view schemaPath,
                        std::size_t depth, Evaluation& evaluation)
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
            const auto errorsBefore = m_errorCount;
            Evaluation propertiesEvaluation;
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
                        Evaluation::Mark(
                            propertiesEvaluation.properties,
                            static_cast<std::size_t>(
                                std::distance(instance.begin(), instance.find(property.key()))));
                    }
                }
            }
            propertiesEvaluation.valid = m_errorCount == errorsBefore;
            evaluation.Merge(propertiesEvaluation);
        }

        std::vector<bool> patternMatches;
        if (const auto* patterns = schema.if_contains("patternProperties"))
        {
            const auto errorsBefore = m_errorCount;
            Evaluation patternsEvaluation;
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
                    const auto* expression = FindPattern(instancePath, patternPath);
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
                            Evaluation::Mark(patternsEvaluation.properties, index);
                        }
                        ++index;
                    }
                }
            }
            patternsEvaluation.valid = m_errorCount == errorsBefore;
            evaluation.Merge(patternsEvaluation);
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
            const auto errorsBefore = m_errorCount;
            Evaluation additionalEvaluation;
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
                Evaluation::Mark(additionalEvaluation.properties, index - 1);
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
            additionalEvaluation.valid = m_errorCount == errorsBefore;
            evaluation.Merge(additionalEvaluation);
        }

        ValidateDependencies(schema, instanceValue, instancePath, schemaPath, depth, evaluation);
        if (m_dialect != JsonSchemaDialect::Draft7)
        {
            if (const auto* unevaluated = schema.if_contains("unevaluatedProperties"))
            {
                const auto keywordPath = ChildPath(schemaPath, "unevaluatedProperties");
                const auto errorsBefore = m_errorCount;
                Evaluation unevaluatedEvaluation;
                std::size_t index = 0;
                for (const auto& property : instance)
                {
                    if (index >= evaluation.properties.size() || !evaluation.properties[index])
                    {
                        Validate(*unevaluated, property.value(),
                                 ChildPath(instancePath, property.key()), keywordPath, depth + 1);
                        Evaluation::Mark(unevaluatedEvaluation.properties, index);
                    }
                    ++index;
                }
                unevaluatedEvaluation.valid = m_errorCount == errorsBefore;
                evaluation.Merge(unevaluatedEvaluation);
            }
        }
    }

    void ValidateDependencies(const JsonSchemaKeywords& schema, const JsonValue& instanceValue,
                              std::string_view instancePath, std::string_view schemaPath,
                              std::size_t depth, Evaluation& evaluation)
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
                    evaluation.Merge(Validate(dependency.value(), instanceValue, instancePath,
                                              dependencyPath, depth + 1));
                }
            }
        }
    }

    [[nodiscard]] std::size_t
    ValidateTupleItems(const JsonArray& schemas, const JsonArray& instance,
                       std::string_view instancePath, std::string_view schemaPath,
                       std::size_t depth, Evaluation& evaluation)
    {
        const auto errorsBefore = m_errorCount;
        Evaluation tupleEvaluation;
        const auto count = std::min(instance.size(), schemas.size());
        for (std::size_t index = 0; index < count; ++index)
        {
            const auto token = std::to_string(index);
            Validate(schemas[index], instance[index], ChildPath(instancePath, token),
                     ChildPath(schemaPath, token), depth + 1);
            Evaluation::Mark(tupleEvaluation.items, index);
        }
        tupleEvaluation.valid = m_errorCount == errorsBefore;
        evaluation.Merge(tupleEvaluation);
        return count;
    }

    void ValidateArray(const JsonSchemaKeywords& schema, const JsonArray& instance,
                       std::string_view instancePath, std::string_view schemaPath,
                       std::size_t depth, Evaluation& evaluation)
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
                                              ChildPath(schemaPath, "prefixItems"), depth,
                                              evaluation);
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
                                              keywordPath, depth, evaluation);
                if (const auto* additional = schema.if_contains("additionalItems"))
                {
                    const auto errorsBefore = m_errorCount;
                    Evaluation additionalEvaluation;
                    const auto additionalPath = ChildPath(schemaPath, "additionalItems");
                    for (std::size_t index = itemStart; index < instance.size(); ++index)
                    {
                        Validate(*additional, instance[index],
                                 ChildPath(instancePath, std::to_string(index)),
                                 additionalPath, depth + 1);
                        Evaluation::Mark(additionalEvaluation.items, index);
                    }
                    additionalEvaluation.valid = m_errorCount == errorsBefore;
                    evaluation.Merge(additionalEvaluation);
                }
            }
            else
            {
                const auto errorsBefore = m_errorCount;
                Evaluation itemsEvaluation;
                for (std::size_t index = itemStart; index < instance.size(); ++index)
                {
                    Validate(*items, instance[index],
                             ChildPath(instancePath, std::to_string(index)), keywordPath,
                             depth + 1);
                    Evaluation::Mark(itemsEvaluation.items, index);
                }
                itemsEvaluation.valid = m_errorCount == errorsBefore;
                evaluation.Merge(itemsEvaluation);
            }
        }

        ValidateContains(schema, instance, instancePath, schemaPath, depth, evaluation);
        if (m_dialect != JsonSchemaDialect::Draft7)
        {
            if (const auto* unevaluated = schema.if_contains("unevaluatedItems"))
            {
                const auto keywordPath = ChildPath(schemaPath, "unevaluatedItems");
                const auto errorsBefore = m_errorCount;
                Evaluation unevaluatedEvaluation;
                for (std::size_t index = 0; index < instance.size(); ++index)
                {
                    if (index >= evaluation.items.size() || !evaluation.items[index])
                    {
                        Validate(*unevaluated, instance[index],
                                 ChildPath(instancePath, std::to_string(index)), keywordPath,
                                 depth + 1);
                        Evaluation::Mark(unevaluatedEvaluation.items, index);
                    }
                }
                unevaluatedEvaluation.valid = m_errorCount == errorsBefore;
                evaluation.Merge(unevaluatedEvaluation);
            }
        }
    }

    void ValidateContains(const JsonSchemaKeywords& schema, const JsonArray& instance,
                          std::string_view instancePath, std::string_view schemaPath,
                          std::size_t depth, Evaluation& evaluation)
    {
        const auto* contains = schema.if_contains("contains");
        if (contains == nullptr)
        {
            return;
        }

        std::size_t matches = 0;
        Evaluation containsEvaluation;
        const auto keywordPath = ChildPath(schemaPath, "contains");
        for (std::size_t index = 0; index < instance.size(); ++index)
        {
            const auto itemMatches =
                EvaluateBranch(*contains, instance[index],
                              ChildPath(instancePath, std::to_string(index)), keywordPath,
                              depth + 1);
            if (!itemMatches)
            {
                return;
            }
            if (itemMatches->valid)
            {
                ++matches;
                Evaluation::Mark(containsEvaluation.items, index);
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
        else if (m_dialect == JsonSchemaDialect::Draft2020_12)
        {
            evaluation.Merge(containsEvaluation);
        }
    }

    void ValidateString(const JsonSchemaKeywords& schema, const JsonString& instance,
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
            const auto* expression = FindPattern(instancePath, keywordPath);
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

    void ValidateNumber(const JsonSchemaKeywords& schema, const JsonValue& instance,
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
            if (!instance.is_double() && !multipleOf->is_double())
            {
                isMultiple = IntegerMagnitude(instance) % IntegerMagnitude(*multipleOf) == 0;
            }
            else
            {
                const auto value = AsDecimal(instance);
                if (!value)
                {
                    AddError(instancePath, keywordPath, value.error());
                    return;
                }
                const auto divisor = AsDecimal(*multipleOf);
                if (!divisor)
                {
                    AddError(instancePath, keywordPath, divisor.error());
                    return;
                }
                isMultiple = IsDecimalMultiple(value.value(), divisor.value());
            }
            if (!isMultiple)
            {
                AddError(instancePath, keywordPath,
                         "number is not a multiple of the required value");
            }
        }
    }

    void ValidateSizeKeyword(const JsonSchemaKeywords& schema, std::string_view keyword,
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

    void ValidateNumberLimit(const JsonSchemaKeywords& schema, std::string_view keyword,
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
    const detail::JsonSchemaReferences& m_references;
    JsonSchemaValidationResult m_result;
    std::size_t m_errorCount = 0;
    std::optional<JsonSchemaValidationError> m_resourceDiagnostic;
    std::optional<JsonSchemaCompileError> m_compileError;
    std::vector<const JsonValue*> m_checkedSchemas;
    std::vector<std::string> m_checkedResources;
    detail::JsonSchemaReferences::Patterns m_patterns;
    std::vector<const std::string*> m_dynamicScope;
    bool m_checkingSchema = false;
    bool m_resourceError = false;
};

[[nodiscard]] Result<detail::JsonSchemaReferences, JsonSchemaCompileError>
CompileReferences(const JsonValue& schema, std::optional<JsonSchemaDialect> dialect,
                  const JsonSchemaCompileOptions& compileOptions)
{
    const JsonSchemaValidationOptions options;
    auto references = detail::JsonSchemaReferences::Compile(
        schema, dialect, options.maxDepth, compileOptions);
    if (!references)
    {
        return Failure(std::move(references.error()));
    }
    const auto& documents = references.value().Documents();
    JsonSchemaValidator validator(references.value().Dialect(), options, documents,
                                  references.value());
    auto patterns = validator.CheckSchema(documents.as_array()[0]);
    if (!patterns)
    {
        return Failure(std::move(patterns.error()));
    }
    references.value().SetPatterns(std::move(patterns.value()));
    return references;
}

} // namespace

JsonSchema::JsonSchema(JsonSchemaDialect dialect,
                       std::shared_ptr<const detail::JsonSchemaReferences> references)
    : m_dialect(dialect), m_references(std::move(references))
{
}

Result<JsonSchema, JsonSchemaCompileError>
JsonSchema::CompileFile(const FilePath& path, JsonSchemaDialect dialect)
{
    return CompileFile(path, dialect, {});
}

Result<JsonSchema, JsonSchemaCompileError>
JsonSchema::CompileFile(const FilePath& path, JsonSchemaDialect dialect,
                        const JsonSchemaCompileOptions& options)
{
    const auto text = File::ReadAllText(path);
    if (!text)
    {
        return Failure(JsonSchemaCompileError{
            JsonSchemaCompileErrorCode::FileReadError,
            dialect,
            {},
            "unable to read schema file: " + path.string(),
            options.retrievalUri,
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
            options.retrievalUri,
        });
    }

    return Compile(schema.value(), dialect, options);
}

Result<JsonSchema, JsonSchemaCompileError>
JsonSchema::Compile(const JsonValue& schema)
{
    return Compile(schema, JsonSchemaCompileOptions{});
}

Result<JsonSchema, JsonSchemaCompileError>
JsonSchema::Compile(const JsonValue& schema, const JsonSchemaCompileOptions& options)
{
    if (!schema.is_object() || !schema.as_object().contains("$schema"))
    {
        return Failure(JsonSchemaCompileError{
            JsonSchemaCompileErrorCode::MissingDialect,
            std::nullopt,
            {},
            "schema does not declare $schema",
            options.retrievalUri,
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
            options.retrievalUri,
        });
    }

    auto references = CompileReferences(schema, std::nullopt, options);
    if (!references)
    {
        return Failure(std::move(references.error()));
    }
    const auto dialect = references.value().Dialect();
    return Success(JsonSchema(
        dialect,
        std::make_shared<const detail::JsonSchemaReferences>(std::move(references.value()))));
}

Result<JsonSchema, JsonSchemaCompileError>
JsonSchema::Compile(const JsonValue& schema, JsonSchemaDialect dialect)
{
    return Compile(schema, dialect, {});
}

Result<JsonSchema, JsonSchemaCompileError>
JsonSchema::Compile(const JsonValue& schema, JsonSchemaDialect dialect,
                    const JsonSchemaCompileOptions& compileOptions)
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
            compileOptions.retrievalUri,
        });
    }

    auto references = CompileReferences(schema, dialect, compileOptions);
    if (!references)
    {
        return Failure(std::move(references.error()));
    }
    return Success(JsonSchema(
        dialect,
        std::make_shared<const detail::JsonSchemaReferences>(std::move(references.value()))));
}

JsonSchemaDialect JsonSchema::Dialect() const noexcept
{
    return m_dialect;
}

JsonSchemaValidationResult
JsonSchema::Validate(const JsonValue& instance,
                     const JsonSchemaValidationOptions& options) const
{
    const auto& documents = m_references->Documents();
    JsonSchemaValidator validator(m_dialect, options, documents, *m_references);
    return validator.ValidateInstance(documents.as_array()[0], instance);
}

} // namespace rad
