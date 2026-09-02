#include <rad/IO/File.h>
#include <rad/IO/Json.h>

#include <boost/json/parse.hpp>
#include <boost/json/serialize.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <utility>

namespace rad
{
namespace
{

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
                        const JsonSchemaValidationOptions& options)
        : m_dialect(dialect), m_options(options)
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

        const auto& object = schema.as_object();
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
                const auto expression = ToStringView(pattern->as_string());
                if (expression.find("\\p{") != std::string_view::npos ||
                    expression.find("\\P{") != std::string_view::npos)
                {
                    AddUnsupportedError(
                        {}, patternPath,
                        "Unicode property escapes are not supported by std::regex");
                }
                else
                {
                    try
                    {
                        static_cast<void>(std::regex(std::string(expression)));
                    }
                    catch (const std::regex_error&)
                    {
                        AddError({}, patternPath,
                                 "pattern is not a valid regular expression");
                    }
                }
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

        if (m_dialect != JsonSchemaDialect::Draft7)
        {
            ValidateDependenciesSchema(object, schemaPath, depth);
        }

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
            if (items->is_array())
            {
                AddUnsupportedError(
                    {}, itemsPath, "tuple validation is not supported by this validator");
            }
            else
            {
                ValidateSchemaDefinition(*items, itemsPath, depth + 1);
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
        if (const auto* dependentRequired = schema.if_contains("dependentRequired"))
        {
            const auto keywordPath = ChildPath(schemaPath, "dependentRequired");
            if (!dependentRequired->is_object())
            {
                AddError({}, keywordPath, "dependentRequired must be an object");
            }
            else
            {
                for (const auto& dependency : dependentRequired->as_object())
                {
                    const auto dependencyPath = ChildPath(keywordPath, dependency.key());
                    if (!dependency.value().is_array())
                    {
                        AddError({}, dependencyPath,
                                 "dependentRequired value must be an array of strings");
                        continue;
                    }
                    if (HasDuplicates(dependency.value().as_array()))
                    {
                        AddError({}, dependencyPath,
                                 "dependent property names must be unique");
                    }
                    for (std::size_t index = 0;
                         index < dependency.value().as_array().size(); ++index)
                    {
                        if (!dependency.value().as_array()[index].is_string())
                        {
                            AddError({}, ChildPath(dependencyPath, std::to_string(index)),
                                     "dependent property name must be a string");
                        }
                    }
                }
            }
        }

        if (const auto* dependentSchemas = schema.if_contains("dependentSchemas"))
        {
            const auto keywordPath = ChildPath(schemaPath, "dependentSchemas");
            if (!dependentSchemas->is_object())
            {
                AddError({}, keywordPath, "dependentSchemas must be an object");
            }
            else
            {
                for (const auto& dependency : dependentSchemas->as_object())
                {
                    ValidateSchemaDefinition(dependency.value(),
                                             ChildPath(keywordPath, dependency.key()),
                                             depth + 1);
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
        JsonSchemaValidator validator(m_dialect, options);
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
        constexpr std::array commonUnsupported = {
            "$ref", "patternProperties", "propertyNames",
        };
        for (const std::string_view keyword : commonUnsupported)
        {
            if (schema.contains(keyword))
            {
                AddUnsupportedError(instancePath, ChildPath(schemaPath, keyword),
                                    "keyword is not supported by this validator");
            }
        }

        if (m_dialect == JsonSchemaDialect::Draft7 && schema.contains("dependencies"))
        {
            AddUnsupportedError(
                instancePath, ChildPath(schemaPath, "dependencies"),
                "keyword is not supported; use dependentRequired or dependentSchemas");
        }

        if (m_dialect != JsonSchemaDialect::Draft7 && schema.contains("$vocabulary"))
        {
            AddUnsupportedError(instancePath, ChildPath(schemaPath, "$vocabulary"),
                                "custom vocabularies are not supported by this validator");
        }

        if (m_dialect != JsonSchemaDialect::Draft2020_12 &&
            schema.contains("additionalItems"))
        {
            AddUnsupportedError(instancePath, ChildPath(schemaPath, "additionalItems"),
                                "tuple validation is not supported by this validator");
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

        if (const auto* additional = schema.if_contains("additionalProperties"))
        {
            const auto keywordPath = ChildPath(schemaPath, "additionalProperties");
            if (!additional->is_bool() && !additional->is_object())
            {
                AddError(instancePath, keywordPath,
                         "additionalProperties must be a boolean or schema");
                return;
            }
            for (const auto& property : instance)
            {
                if (properties != nullptr && properties->contains(property.key()))
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

        if (m_dialect != JsonSchemaDialect::Draft7)
        {
            ValidateDependencies(schema, instanceValue, instancePath, schemaPath, depth);
        }
    }

    void ValidateDependencies(const JsonObject& schema, const JsonValue& instanceValue,
                              std::string_view instancePath, std::string_view schemaPath,
                              std::size_t depth)
    {
        const auto& instance = instanceValue.as_object();
        if (const auto* dependentRequired = schema.if_contains("dependentRequired");
            dependentRequired != nullptr && dependentRequired->is_object())
        {
            const auto keywordPath = ChildPath(schemaPath, "dependentRequired");
            for (const auto& dependency : dependentRequired->as_object())
            {
                if (!instance.contains(dependency.key()))
                {
                    continue;
                }
                for (const auto& required : dependency.value().as_array())
                {
                    const auto requiredName = ToStringView(required.as_string());
                    if (!instance.contains(requiredName))
                    {
                        AddError(ChildPath(instancePath, requiredName),
                                 ChildPath(keywordPath, dependency.key()),
                                 "dependent property is missing");
                    }
                }
            }
        }

        if (const auto* dependentSchemas = schema.if_contains("dependentSchemas");
            dependentSchemas != nullptr && dependentSchemas->is_object())
        {
            const auto keywordPath = ChildPath(schemaPath, "dependentSchemas");
            for (const auto& dependency : dependentSchemas->as_object())
            {
                if (instance.contains(dependency.key()))
                {
                    Validate(dependency.value(), instanceValue, instancePath,
                             ChildPath(keywordPath, dependency.key()), depth + 1);
                }
            }
        }
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
                itemStart = std::min(instance.size(), prefixItems->as_array().size());
                const auto keywordPath = ChildPath(schemaPath, "prefixItems");
                for (std::size_t index = 0; index < itemStart; ++index)
                {
                    Validate(prefixItems->as_array()[index], instance[index],
                             ChildPath(instancePath, std::to_string(index)),
                             ChildPath(keywordPath, std::to_string(index)), depth + 1);
                }
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
                AddUnsupportedError(
                    instancePath, keywordPath,
                    "tuple validation is not supported by this validator");
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
            try
            {
                const std::regex expression(std::string(ToStringView(pattern->as_string())));
                if (!std::regex_search(value.begin(), value.end(), expression))
                {
                    AddError(instancePath, keywordPath,
                             "string does not match the required pattern");
                }
            }
            catch (const std::regex_error&)
            {
                AddError(instancePath, keywordPath, "pattern is not a valid regular expression");
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
    JsonSchemaValidationResult m_result;
    std::optional<JsonSchemaCompileError> m_compileError;
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
    JsonSchemaValidator validator(dialect, options);
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
    JsonSchemaValidator validator(m_dialect, options);
    return validator.ValidateInstance(m_schema, instance);
}

} // namespace rad
