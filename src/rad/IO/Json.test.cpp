#include <rad/IO/File.h>
#include <rad/IO/Json.h>
#include <rad/Core/Unicode.h>

#include <boost/json.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{

[[nodiscard]] std::string
FormatValidationErrors(const rad::JsonSchemaValidationResult& result)
{
    std::string output;
    for (const auto& error : result.errors)
    {
        output += std::format("instance={}, schema={}: {}\n", error.instancePath,
                              error.schemaPath, error.message);
    }
    return output;
}

[[nodiscard]] bool IsKnownInvalidOfficialSchema(const rad::JsonValue& schema)
{
    if (!schema.is_object())
    {
        return false;
    }
    const auto* enumValue = schema.as_object().if_contains("enum");
    return enumValue != nullptr && enumValue->is_array() && enumValue->as_array().empty();
}

} // namespace

static_assert(std::is_same_v<rad::JsonValue, boost::json::value>);
static_assert(std::is_same_v<rad::JsonObject, boost::json::object>);
static_assert(std::is_same_v<rad::JsonArray, boost::json::array>);
static_assert(std::is_same_v<rad::JsonString, boost::json::string>);
static_assert(std::is_same_v<rad::JsonKind, boost::json::kind>);
static_assert(std::is_same_v<rad::JsonErrorCode, boost::system::error_code>);
static_assert(std::is_same_v<rad::JsonParseOptions, boost::json::parse_options>);

TEST(IO, ParseJson)
{
    // Parse Non-Standard JSON with Comments and Trailing Commas
    {
        constexpr std::string_view jsonString = R"json(
{
  // Comments and trailing commas are useful in configuration files.
  "name": "config",
  "values": [1, 2, 3,],
}
)json";

        rad::JsonParseOptions options;
        options.allow_comments = true;
        options.allow_trailing_commas = true;
        const auto json = rad::ParseJson(jsonString, options);
        ASSERT_TRUE(json) << json.error().message();
        EXPECT_EQ(json.value().as_object().at("values").as_array().size(), 3);
    }
}

TEST(IO, JsonSchemaExamples)
{
    // From https://json-schema.org/learn/miscellaneous-examples.
    // The arrays example is omitted because it requires unsupported $ref resolution.
    struct Example
    {
        std::string_view name;
        std::string_view schema;
        std::string_view data;
        bool valid;
    };

    constexpr Example examples[] = {
        {
            "basic person",
            R"json({
  "$id": "https://example.com/person.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Person",
  "type": "object",
  "properties": {
    "firstName": {
      "type": "string",
      "description": "The person's first name."
    },
    "lastName": {
      "type": "string",
      "description": "The person's last name."
    },
    "age": {
      "description": "Age in years which must be equal to or greater than zero.",
      "type": "integer",
      "minimum": 0
    }
  }
})json",
            R"json({"firstName": "John", "lastName": "Doe", "age": 21})json",
            true,
        },
        {
            "enumerated values",
            R"json({
  "$id": "https://example.com/enumerated-values.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Enumerated Values",
  "type": "object",
  "properties": {
    "data": {"enum": [42, true, "hello", null, [1, 2, 3]]}
  }
})json",
            R"json({"data": [1, 2, 3]})json",
            true,
        },
        {
            "regular expression",
            R"json({
  "$id": "https://example.com/regex-pattern.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Regular Expression Pattern",
  "type": "object",
  "properties": {
    "code": {"type": "string", "pattern": "^[A-Z]{3}-\\d{3}$"}
  }
})json",
            R"json({"code": "ABC-123"})json",
            true,
        },
        {
            "complex nested object",
            R"json({
  "$id": "https://example.com/complex-object.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Complex Object",
  "type": "object",
  "required": ["name", "age"],
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer", "minimum": 0},
    "address": {
      "type": "object",
      "required": ["street", "city", "state", "postalCode"],
      "properties": {
        "street": {"type": "string"},
        "city": {"type": "string"},
        "state": {"type": "string"},
        "postalCode": {"type": "string", "pattern": "\\d{5}"}
      }
    },
    "hobbies": {"type": "array", "items": {"type": "string"}}
  }
})json",
            R"json({
  "name": "John Doe",
  "age": 25,
  "address": {
    "street": "123 Main St",
    "city": "New York",
    "state": "NY",
    "postalCode": "10001"
  },
  "hobbies": ["reading", "running"]
})json",
            true,
        },
        {
            "conditional dependent required",
            R"json({
  "$id": "https://example.com/conditional-validation-dependentRequired.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Conditional Validation with dependentRequired",
  "type": "object",
  "properties": {
    "foo": {"type": "boolean"},
    "bar": {"type": "string"}
  },
  "dependentRequired": {"foo": ["bar"]}
})json",
            R"json({"foo": true, "bar": "Hello World"})json",
            true,
        },
        {
            "conditional dependent required without either property",
            R"json({
  "$id": "https://example.com/conditional-validation-dependentRequired.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Conditional Validation with dependentRequired",
  "type": "object",
  "properties": {
    "foo": {"type": "boolean"},
    "bar": {"type": "string"}
  },
  "dependentRequired": {"foo": ["bar"]}
})json",
            R"json({})json",
            true,
        },
        {
            "conditional dependent required missing property",
            R"json({
  "$id": "https://example.com/conditional-validation-dependentRequired.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Conditional Validation with dependentRequired",
  "type": "object",
  "properties": {
    "foo": {"type": "boolean"},
    "bar": {"type": "string"}
  },
  "dependentRequired": {"foo": ["bar"]}
})json",
            R"json({"foo": true})json",
            false,
        },
        {
            "conditional dependent schema",
            R"json({
  "$id": "https://example.com/conditional-validation-dependentSchemas.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Conditional Validation with dependentSchemas",
  "type": "object",
  "properties": {
    "foo": {"type": "boolean"},
    "propertiesCount": {"type": "integer", "minimum": 0}
  },
  "dependentSchemas": {
    "foo": {
      "required": ["propertiesCount"],
      "properties": {"propertiesCount": {"minimum": 7}}
    }
  }
})json",
            R"json({"foo": true, "propertiesCount": 10})json",
            true,
        },
        {
            "conditional dependent schema without triggering property",
            R"json({
  "$id": "https://example.com/conditional-validation-dependentSchemas.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Conditional Validation with dependentSchemas",
  "type": "object",
  "properties": {
    "foo": {"type": "boolean"},
    "propertiesCount": {"type": "integer", "minimum": 0}
  },
  "dependentSchemas": {
    "foo": {
      "required": ["propertiesCount"],
      "properties": {"propertiesCount": {"minimum": 7}}
    }
  }
})json",
            R"json({"propertiesCount": 5})json",
            true,
        },
        {
            "conditional dependent schema below conditional minimum",
            R"json({
  "$id": "https://example.com/conditional-validation-dependentSchemas.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Conditional Validation with dependentSchemas",
  "type": "object",
  "properties": {
    "foo": {"type": "boolean"},
    "propertiesCount": {"type": "integer", "minimum": 0}
  },
  "dependentSchemas": {
    "foo": {
      "required": ["propertiesCount"],
      "properties": {"propertiesCount": {"minimum": 7}}
    }
  }
})json",
            R"json({"foo": true, "propertiesCount": 5})json",
            false,
        },
        {
            "conditional member",
            R"json({
  "$id": "https://example.com/conditional-validation-if-else.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Conditional Validation with If-Else",
  "type": "object",
  "required": ["isMember"],
  "properties": {
    "isMember": {"type": "boolean"},
    "membershipNumber": {"type": "string"}
  },
  "if": {"properties": {"isMember": {"const": true}}},
  "then": {
    "properties": {
      "membershipNumber": {"type": "string", "minLength": 10, "maxLength": 10}
    }
  },
  "else": {
    "properties": {
      "membershipNumber": {"type": "string", "minLength": 15}
    }
  }
})json",
            R"json({"isMember": true, "membershipNumber": "1234567890"})json",
            true,
        },
        {
            "conditional guest",
            R"json({
  "$id": "https://example.com/conditional-validation-if-else.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Conditional Validation with If-Else",
  "type": "object",
  "required": ["isMember"],
  "properties": {
    "isMember": {"type": "boolean"},
    "membershipNumber": {"type": "string"}
  },
  "if": {"properties": {"isMember": {"const": true}}},
  "then": {
    "properties": {
      "membershipNumber": {"type": "string", "minLength": 10, "maxLength": 10}
    }
  },
  "else": {
    "properties": {
      "membershipNumber": {"type": "string", "minLength": 15}
    }
  }
})json",
            R"json({"isMember": false, "membershipNumber": "GUEST1234567890"})json",
            true,
        },
    };

    for (const auto& example : examples)
    {
        SCOPED_TRACE(example.name);
        const auto schema = rad::ParseJson(example.schema);
        ASSERT_TRUE(schema) << schema.error().message();
        const auto data = rad::ParseJson(example.data);
        ASSERT_TRUE(data) << data.error().message();

        const auto compiled = rad::JsonSchema::Compile(schema.value());
        ASSERT_TRUE(compiled) << compiled.error().schemaPath << ": "
                              << compiled.error().message;
        const auto result = compiled.value().Validate(data.value());
        EXPECT_EQ(static_cast<bool>(result), example.valid)
            << FormatValidationErrors(result);
    }
}

TEST(IO, JsonSchemaReferenceDiagnosticsAndLimits)
{
    const auto schema = rad::ParseJson(
        R"json({"$defs": {"node": {"type": "object", "properties":
                 {"next": {"$ref": "#/$defs/node"}}}},
                 "$ref": "#/$defs/node"})json");
    ASSERT_TRUE(schema);
    const auto compiled = rad::JsonSchema::Compile(
        schema.value(), rad::JsonSchemaDialect::Draft2020_12);
    ASSERT_TRUE(compiled) << compiled.error().message;
    const auto invalid = compiled.value().Validate(rad::JsonObject{
        {"next", rad::JsonObject{{"next", 12}}}});
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.errors[0].instancePath, "/next/next");
    EXPECT_EQ(invalid.errors[0].schemaPath, "/$defs/node/type");

    rad::JsonSchemaValidationOptions options;
    options.maxDepth = 2;
    const auto limited = compiled.value().Validate(rad::JsonObject{
        {"next", rad::JsonObject{{"next", rad::JsonObject{}}}}}, options);
    ASSERT_FALSE(limited);
    EXPECT_NE(limited.errors[0].message.find("maximum validation depth"),
              std::string::npos);

    const auto selfReference = rad::ParseJson(R"json({"$ref": "#"})json");
    ASSERT_TRUE(selfReference);
    const auto selfCompiled = rad::JsonSchema::Compile(
        selfReference.value(), rad::JsonSchemaDialect::Draft2020_12);
    ASSERT_TRUE(selfCompiled);
    options.maxErrors = 1;
    const auto selfResult = selfCompiled.value().Validate(1, options);
    ASSERT_EQ(selfResult.errors.size(), 1);
    EXPECT_NE(selfResult.errors[0].message.find("maximum validation depth"),
              std::string::npos);
}

TEST(IO, JsonSchemaReferenceResourceLimitsAndCopies)
{
    const auto nestedSource = rad::ParseJson(
        R"json({"$defs": {"inner": {"$id": "sub.json", "$ref": "#"}},
                 "$ref": "#/$defs/inner"})json");
    ASSERT_TRUE(nestedSource);
    const auto nestedCompiled =
        rad::JsonSchema::Compile(nestedSource.value(), rad::JsonSchemaDialect::Draft2020_12);
    ASSERT_TRUE(nestedCompiled) << nestedCompiled.error().message;
    rad::JsonSchemaValidationOptions options;
    options.maxDepth = 2;
    const auto recursive = nestedCompiled.value().Validate(1, options);
    ASSERT_FALSE(recursive);
    ASSERT_EQ(recursive.errors.size(), 1);
    EXPECT_EQ(recursive.errors[0].schemaPath, "/$defs/inner");
    EXPECT_EQ(recursive.errors[0].message, "maximum validation depth exceeded");

    const auto makeSchema = []
    {
        const auto schema = rad::ParseJson(
            R"json({"$defs": {"inner": {"$id": "sub.json", "type": "string"}},
                     "$ref": "sub.json"})json");
        return rad::JsonSchema::Compile(schema.value(), rad::JsonSchemaDialect::Draft2020_12);
    };
    const auto compiled = makeSchema();
    ASSERT_TRUE(compiled) << compiled.error().message;
    auto copied = compiled.value();
    auto moved = std::move(copied);
    const auto invalid = moved.Validate(1);
    ASSERT_FALSE(invalid);
    ASSERT_EQ(invalid.errors.size(), 1);
    EXPECT_EQ(invalid.errors[0].schemaPath, "/$defs/inner/type");
}

TEST(IO, JsonSchemaResourceIdentifierDiagnostics)
{
    struct Case
    {
        std::string_view schema;
        std::string_view path;
    };
    constexpr Case cases[] = {
        {R"json({"$id": 42})json", "/$id"},
        {R"json({"$id": "https://example.com/%ZZ"})json", "/$id"},
        {R"json({"$id": "#name"})json", "/$id"},
        {R"json({"$anchor": 42})json", "/$anchor"},
        {R"json({"$anchor": "1invalid"})json", "/$anchor"},
        {R"json({"$defs": {"a": {"$anchor": "same"}, "b": {"$anchor": "same"}}})json",
         "/$defs/b/$anchor"},
        {R"json({"$defs": {"a": {"$id": "https://example.com/a"},
                           "b": {"$id": "https://EXAMPLE.COM/%61"}}})json",
         "/$defs/b/$id"},
    };
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        for (const auto& testCase : cases)
        {
            SCOPED_TRACE(testCase.schema);
            const auto schema = rad::ParseJson(testCase.schema);
            ASSERT_TRUE(schema);
            const auto compiled = rad::JsonSchema::Compile(schema.value(), dialect);
            ASSERT_FALSE(compiled);
            EXPECT_EQ(compiled.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
            EXPECT_EQ(compiled.error().schemaPath, testCase.path);
        }

        const auto deferred = rad::ParseJson(
            R"json({"allOf": [{"$ref": "child#target"}, {"$ref": "#/storage"}],
                     "storage": {"$id": "child", "$anchor": "target",
                                 "$defs": {"value": {"type": "string"}},
                                 "$ref": "#/$defs/value"}})json");
        ASSERT_TRUE(deferred);
        const auto compiled = rad::JsonSchema::Compile(deferred.value(), dialect);
        ASSERT_TRUE(compiled) << compiled.error().schemaPath << ": " << compiled.error().message;
        const auto result = compiled.value().Validate(1);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.errors[0].schemaPath, "/storage/$defs/value/type");

        const auto annotations = rad::ParseJson(
            R"json({"const": {"$id": 42, "$anchor": "1invalid"},
                     "examples": [{"$id": "hidden", "$ref": "external"}]})json");
        ASSERT_TRUE(annotations);
        EXPECT_TRUE(rad::JsonSchema::Compile(annotations.value(), dialect));
    }
    const auto ignored = rad::ParseJson(
        R"json({"$ref": "#/definitions/value", "$id": 42,
                 "definitions": {"value": true, "unused": {"$ref": "external.json"},
                                 "malformed": {"$ref": 42}}})json");
    ASSERT_TRUE(ignored);
    EXPECT_TRUE(rad::JsonSchema::Compile(ignored.value(), rad::JsonSchemaDialect::Draft7));
}

TEST(IO, JsonSchemaReferenceDiscoveryOrder)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        for (const bool descendantFirst : {true, false})
        {
            SCOPED_TRACE(descendantFirst);
            constexpr std::string_view schemas[] = {
                R"json({"allOf": [{"$ref": "#/storage/$defs/use"}, {"$ref": "#/storage"}],
                         "storage": {"$id": "child",
                                     "$defs": {"use": {"$ref": "#/$defs/value"},
                                               "value": {"type": "integer"}}}})json",
                R"json({"allOf": [{"$ref": "#/storage/$defs/use"}, {"$ref": "#/entry"},
                                 {"$ref": "child/use.json#use"}],
                         "entry": {"$ref": "#/storage"},
                         "storage": {"$id": "child/",
                                     "$defs": {"use": {"$id": "use.json", "$anchor": "use",
                                                       "$ref": "value.json"},
                                               "value": {"$id": "value.json",
                                                         "type": "integer"}}}})json",
            };
            for (const auto text : schemas)
            {
                SCOPED_TRACE(text);
                const auto schema = rad::ParseJson(text);
                ASSERT_TRUE(schema);
                auto value = schema.value();
                if (!descendantFirst)
                {
                    auto& branches = value.as_object().at("allOf").as_array();
                    std::swap(branches[0], branches[1]);
                }
                const auto compiled = rad::JsonSchema::Compile(value, dialect);
                ASSERT_TRUE(compiled) << compiled.error().schemaPath << ": "
                                      << compiled.error().message;
                EXPECT_TRUE(compiled.value().Validate(1));
                const auto invalid = compiled.value().Validate("invalid");
                ASSERT_FALSE(invalid);
                rad::JsonSchemaValidationOptions options;
                options.maxErrors = 1;
                const auto limited = compiled.value().Validate("invalid", options);
                ASSERT_FALSE(limited);
                ASSERT_EQ(limited.errors.size(), 1);
                EXPECT_EQ(limited.errors[0].schemaPath, "/storage/$defs/value/type");
            }
        }
    }
}

TEST(IO, JsonSchemaInvalidReferences)
{
    struct Case
    {
        std::string_view reference;
        rad::JsonSchemaCompileErrorCode code;
    };
    constexpr Case cases[] = {
        {R"json(7)json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("#/missing")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("#/a~2b")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("#/%ZZ")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("#/list/01")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("#/list/-")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("#/list/1")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("other.json#/$defs/item")json",
         rad::JsonSchemaCompileErrorCode::UnsupportedFeature},
        {R"json("#named")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
    };
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.reference);
        const auto schema = rad::ParseJson(
            std::format(R"json({{"list": [{{"type": "string"}}], "$ref": {}}})json",
                        testCase.reference));
        ASSERT_TRUE(schema);
        const auto compiled = rad::JsonSchema::Compile(
            schema.value(), rad::JsonSchemaDialect::Draft2020_12);
        ASSERT_FALSE(compiled);
        EXPECT_EQ(compiled.error().code, testCase.code);
        EXPECT_EQ(compiled.error().schemaPath, "/$ref");
    }

    const auto nonSchemaTarget = rad::ParseJson(
        R"json({"list": [{"type": "string"}], "$ref": "#/list/0/type"})json");
    ASSERT_TRUE(nonSchemaTarget);
    const auto invalidTarget = rad::JsonSchema::Compile(
        nonSchemaTarget.value(), rad::JsonSchemaDialect::Draft2020_12);
    ASSERT_FALSE(invalidTarget);
    EXPECT_EQ(invalidTarget.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    EXPECT_EQ(invalidTarget.error().schemaPath, "/list/0/type");
}

#if !defined(RAD_JSON_SCHEMA_USE_STD_REGEX) || !RAD_JSON_SCHEMA_USE_STD_REGEX
TEST(IO, JsonSchemaRegexWhitespaceCompatibility)
{
    struct Pattern
    {
        std::string_view text;
        bool complement;
        bool includeUnderscore;
    };
    constexpr Pattern patterns[] = {
        {R"(^\s$)", false, false},
        {R"(^\S$)", true, false},
        {R"(^[\s]$)", false, false},
        {R"(^[^\S]$)", false, false},
        {R"(^[\S]$)", true, false},
        {R"(^[^\s]$)", true, false},
        {R"(^[\s_]$)", false, true},
        {R"(^[_\S]$)", true, true},
        {R"(^[\s\S]$)", true, true},
    };
    constexpr char32_t boundaries[] = {
        0x0009, 0x000d, 0x0020, 0x00a0, 0x1680, 0x2000, 0x200a,
        0x2028, 0x2029, 0x202f, 0x205f, 0x3000, 0xfeff,
    };
    std::vector<char32_t> points = {0, 0x0085, 0x180e, U'_', U'a',
                                   0xd7ff, 0xe000, 0x10000, 0x10ffff};
    for (const char32_t boundary : boundaries)
    {
        points.push_back(boundary - 1);
        points.push_back(boundary);
        points.push_back(boundary + 1);
    }
    const auto isWhitespace = [](char32_t point) {
        return (point >= 0x0009 && point <= 0x000d) || point == 0x0020 ||
               point == 0x00a0 || point == 0x1680 ||
               (point >= 0x2000 && point <= 0x200a) ||
               point == 0x2028 || point == 0x2029 || point == 0x202f ||
               point == 0x205f || point == 0x3000 || point == 0xfeff;
    };
    for (const auto& pattern : patterns)
    {
        SCOPED_TRACE(pattern.text);
        const auto compiled = rad::JsonSchema::Compile(
            rad::JsonObject{{"pattern", pattern.text}},
            rad::JsonSchemaDialect::Draft2020_12);
        ASSERT_TRUE(compiled) << compiled.error().message;
        for (const char32_t point : points)
        {
            SCOPED_TRACE(static_cast<std::uint32_t>(point));
            const bool expected = pattern.text == R"(^[\s\S]$)" ||
                                  (isWhitespace(point) != pattern.complement) ||
                                  (pattern.includeUnderscore && point == U'_');
            const auto result = compiled.value().Validate(
                rad::JsonValue(rad::Utf32ToUtf8(std::u32string_view(&point, 1))));
            EXPECT_EQ(static_cast<bool>(result), expected) << FormatValidationErrors(result);
        }
    }

    struct Literal
    {
        std::string_view pattern;
        std::string_view value;
    };
    constexpr Literal literals[] = {
        {R"(^\\s$)", R"(\s)"},
        {R"(^\Q\s\S\E$)", R"(\s\S)"},
        {R"(^[\s\-]$)", "-"},
        {R"(^[-\s]$)", "-"},
        {R"(^[^\-\S]$)", " "},
        {R"(^\[\s\]$)", "[ ]"},
        {R"(^[[:digit:]\s]$)", " "},
        {R"(^(?# [\s)\s$)", " "},
        {R"(^\c[\s$)", "\x1b "},
    };
    for (const auto& literal : literals)
    {
        SCOPED_TRACE(literal.pattern);
        const auto compiled = rad::JsonSchema::Compile(
            rad::JsonObject{{"pattern", literal.pattern}},
            rad::JsonSchemaDialect::Draft2020_12);
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(rad::JsonValue(literal.value)));
    }
    for (const auto* pattern : {R"([a-\s])", R"([\S-a])", R"([\s-\S])"})
    {
        SCOPED_TRACE(pattern);
        const auto compiled = rad::JsonSchema::Compile(
            rad::JsonObject{{"pattern", pattern}},
            rad::JsonSchemaDialect::Draft2020_12);
        ASSERT_FALSE(compiled);
        EXPECT_EQ(compiled.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    }
}
#endif

TEST(IO, JsonSchemaResourceFailureWithErrorLimit)
{
    const auto schema = rad::ParseJson(
        R"json({"anyOf": [
            {"type": "integer", "properties": {"child": {"allOf": [true]}}},
            true
        ]})json");
    ASSERT_TRUE(schema);
    const auto compiled = rad::JsonSchema::Compile(
        schema.value(), rad::JsonSchemaDialect::Draft2020_12);
    ASSERT_TRUE(compiled) << compiled.error().message;

    rad::JsonSchemaValidationOptions options;
    options.maxDepth = 2;
    for (const std::size_t maxErrors : {0U, 1U, 2U})
    {
        SCOPED_TRACE(maxErrors);
        options.maxErrors = maxErrors;
        const auto result = compiled.value().Validate(
            rad::JsonObject{{"child", rad::JsonObject{}}}, options);
        ASSERT_FALSE(result);
        ASSERT_EQ(result.errors.size(), 1);
        EXPECT_EQ(result.errors[0].instancePath, "/child");
        EXPECT_EQ(result.errors[0].schemaPath, "/anyOf/0/properties/child/allOf/0");
        EXPECT_EQ(result.errors[0].message, "maximum validation depth exceeded");
    }
    options.maxDepth = 3;
    EXPECT_TRUE(compiled.value().Validate(
        rad::JsonObject{{"child", rad::JsonObject{}}}, options));
}

TEST(IO, JsonSchemaUnevaluatedItemsDiagnostics)
{
    const auto schema = rad::ParseJson(R"json({"unevaluatedItems": {"type": "integer"}})json");
    ASSERT_TRUE(schema);
    for (const auto dialect : {rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const auto compiled = rad::JsonSchema::Compile(schema.value(), dialect);
        ASSERT_TRUE(compiled) << compiled.error().message;
        const auto result = compiled.value().Validate(rad::JsonArray{1, "invalid"});
        ASSERT_FALSE(result);
        ASSERT_EQ(result.errors.size(), 1);
        EXPECT_EQ(result.errors[0].instancePath, "/1");
        EXPECT_EQ(result.errors[0].schemaPath, "/unevaluatedItems/type");

        rad::JsonSchemaValidationOptions options;
        options.maxDepth = 0;
        options.maxErrors = 1;
        const auto limited = compiled.value().Validate(rad::JsonArray{1}, options);
        ASSERT_FALSE(limited);
        ASSERT_EQ(limited.errors.size(), 1);
        EXPECT_EQ(limited.errors[0].instancePath, "/0");
        EXPECT_EQ(limited.errors[0].schemaPath, "/unevaluatedItems");
        EXPECT_EQ(limited.errors[0].message, "maximum validation depth exceeded");

        const auto invalid = rad::JsonSchema::Compile(
            rad::JsonObject{{"unevaluatedItems", 42}}, dialect);
        ASSERT_FALSE(invalid);
        EXPECT_EQ(invalid.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
        EXPECT_EQ(invalid.error().schemaPath, "/unevaluatedItems");
    }
    const auto olderDraft = rad::JsonSchema::Compile(
        rad::JsonObject{{"unevaluatedItems", 42}}, rad::JsonSchemaDialect::Draft7);
    ASSERT_TRUE(olderDraft) << olderDraft.error().message;
    EXPECT_TRUE(olderDraft.value().Validate(rad::JsonArray{1, "ignored"}));
}

TEST(IO, JsonSchemaOfficialTestSuite)
{
    const char* suitePath = std::getenv("JSON_SCHEMA_TEST_SUITE");
    if (suitePath == nullptr || *suitePath == '\0')
    {
        GTEST_LOG_(WARNING) << "JSON_SCHEMA_TEST_SUITE is not specified";
        GTEST_SKIP();
    }

    const std::filesystem::path suiteRoot = suitePath;
    if (!std::filesystem::is_directory(suiteRoot))
    {
        GTEST_LOG_(WARNING) << "JSON_SCHEMA_TEST_SUITE does not exist: "
                            << suiteRoot.string();
        GTEST_SKIP();
    }

    struct Suite
    {
        rad::JsonSchemaDialect dialect;
        std::string_view directory;
    };
    constexpr Suite suites[] = {
        {rad::JsonSchemaDialect::Draft7, "draft7"},
        {rad::JsonSchemaDialect::Draft2019_09, "draft2019-09"},
        {rad::JsonSchemaDialect::Draft2020_12, "draft2020-12"},
    };

    std::size_t availableSuites = 0;
    std::size_t executedCases = 0;
    for (const auto& suite : suites)
    {
        const auto testsDirectory = suiteRoot / "tests" / suite.directory;
        if (!std::filesystem::is_directory(testsDirectory))
        {
            GTEST_LOG_(WARNING) << "Schema tests not found under "
                                << testsDirectory.string();
            continue;
        }
        ++availableSuites;
        std::size_t suiteExecutedCases = 0;
        std::size_t suiteReferenceCases = 0;
        std::size_t failedReferenceGroups = 0;

        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(testsDirectory))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".json")
            {
                files.push_back(entry.path());
            }
        }
        for (const auto* name : {"ecmascript-regex.json", "non-bmp-regex.json"})
        {
            const auto file = testsDirectory / "optional" / name;
            if (std::filesystem::is_regular_file(file))
            {
                files.push_back(file);
            }
        }
        std::sort(files.begin(), files.end());

        for (const auto& file : files)
        {
            SCOPED_TRACE(file.string());
            const bool referenceFile = file.filename() == "ref.json";
            const auto text = rad::File::ReadAllText(file);
            if (!text)
            {
                ADD_FAILURE() << "Unable to read test file";
                continue;
            }
            const auto document = rad::ParseJson(*text);
            if (!document || !document.value().is_array())
            {
                ADD_FAILURE() << "Unable to parse test file";
                continue;
            }

            for (const auto& groupValue : document.value().as_array())
            {
                ASSERT_TRUE(groupValue.is_object());
                const auto& group = groupValue.as_object();
                const auto& description = group.at("description").as_string();
                SCOPED_TRACE(std::string(description.data(), description.size()));

                const auto compiled =
                    rad::JsonSchema::Compile(group.at("schema"), suite.dialect);
                if (IsKnownInvalidOfficialSchema(group.at("schema")))
                {
                    if (compiled)
                    {
                        ADD_FAILURE() << "empty enum schema must be rejected";
                    }
                    else
                    {
                        EXPECT_EQ(compiled.error().code,
                                  rad::JsonSchemaCompileErrorCode::InvalidSchema);
                        EXPECT_EQ(compiled.error().schemaPath, "/enum");
                    }
                    continue;
                }
                if (!compiled)
                {
                    if (referenceFile)
                    {
                        ++failedReferenceGroups;
                    }
                    ADD_FAILURE() << compiled.error().schemaPath << ": "
                                  << compiled.error().message;
                    continue;
                }

                for (const auto& testValue : group.at("tests").as_array())
                {
                    const auto& test = testValue.as_object();
                    const auto& testDescription = test.at("description").as_string();
                    SCOPED_TRACE(
                        std::string(testDescription.data(), testDescription.size()));

                    const bool expected = test.at("valid").as_bool();
                    const auto result = compiled.value().Validate(test.at("data"));
                    ++suiteExecutedCases;
                    ++executedCases;
                    if (referenceFile)
                    {
                        ++suiteReferenceCases;
                    }
                    EXPECT_EQ(static_cast<bool>(result), expected)
                        << std::format("data: {}\n{}",
                                       rad::PrettyJson(test.at("data")),
                                       FormatValidationErrors(result));
                    rad::JsonSchemaValidationOptions limitedOptions;
                    limitedOptions.maxErrors = 1;
                    const auto limited = compiled.value().Validate(test.at("data"), limitedOptions);
                    EXPECT_EQ(static_cast<bool>(limited), expected)
                        << FormatValidationErrors(limited);
                    EXPECT_LE(limited.errors.size(), 1);
                }
            }
        }
        EXPECT_GT(suiteExecutedCases, 0) << testsDirectory.string();
        EXPECT_GT(suiteReferenceCases, 0) << testsDirectory.string() << "\\ref.json";
        GTEST_LOG_(INFO) << suite.directory << ": " << suiteReferenceCases
                         << " reference cases executed, " << failedReferenceGroups
                         << " reference groups failed to compile";
    }

    if (availableSuites == 0)
    {
        GTEST_SKIP() << "No supported schema test directories were found";
    }
    EXPECT_GT(executedCases, 0);
}
