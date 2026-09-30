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
#include <cmath>
#include <limits>
#include <future>

namespace
{

[[nodiscard]] std::string
FormatValidationErrors(const rad::JsonSchemaValidationResult& result)
{
    std::string output;
    for (const auto& error : result.errors)
    {
        output += std::format("instance={}, document={}, schema={}: {}\n", error.instancePath,
                              error.schemaUri, error.schemaPath, error.message);
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

TEST(IO, JsonSchemaRegexCacheLifetimeAndConcurrency)
{
    const auto makeSchema = []
    {
        rad::JsonSchemaCompileOptions options;
        options.retrievalUri = "https://example.com/root.json";
        options.documents = {
            {"https://example.com/remote.json", rad::JsonObject{{"pattern", "^remote$"}}},
        };
        return rad::JsonSchema::Compile(
            rad::JsonObject{
                {"properties", rad::JsonObject{
                    {"local", rad::JsonObject{{"pattern", "^local$"}}},
                    {"remote", rad::JsonObject{{"$ref", "remote.json"}}}}},
                {"patternProperties", rad::JsonObject{
                    {"^x/~", rad::JsonObject{{"pattern", "^extra$"}}}}},
            },
            rad::JsonSchemaDialect::Draft2020_12, options);
    };
    const auto compiled = makeSchema();
    ASSERT_TRUE(compiled) << compiled.error().message;
    auto copied = compiled.value();
    auto moved = std::move(copied);
    const rad::JsonValue valid =
        rad::JsonObject{{"local", "local"}, {"remote", "remote"}, {"x/~tag", "extra"}};
    const rad::JsonValue invalid =
        rad::JsonObject{{"local", "remote"}, {"remote", "local"}, {"x/~tag", "bad"}};
    const auto result = moved.Validate(invalid);
    ASSERT_EQ(result.errors.size(), 3);
    EXPECT_EQ(result.errors[0].schemaPath, "/properties/local/pattern");
    EXPECT_EQ(result.errors[0].schemaUri, "https://example.com/root.json");
    EXPECT_EQ(result.errors[1].schemaPath, "/pattern");
    EXPECT_EQ(result.errors[1].schemaUri, "https://example.com/remote.json");
    EXPECT_EQ(result.errors[2].schemaPath, "/patternProperties/^x~1~0/pattern");
    EXPECT_EQ(result.errors[2].instancePath, "/x~1~0tag");

    std::vector<std::future<bool>> workers;
    for (std::size_t worker = 0; worker < 4; ++worker)
    {
        workers.push_back(std::async(std::launch::async, [schema = moved, valid, invalid]
        {
            for (std::size_t iteration = 0; iteration < 16; ++iteration)
            {
                if (!schema.Validate(valid) || schema.Validate(invalid).errors.size() != 3)
                {
                    return false;
                }
            }
            return true;
        }));
    }
    for (auto& worker : workers)
    {
        EXPECT_TRUE(worker.get());
    }
}

#if !defined(RAD_JSON_SCHEMA_USE_STD_REGEX) || !RAD_JSON_SCHEMA_USE_STD_REGEX
TEST(IO, JsonSchemaRegexMatchLimits)
{
    constexpr std::string_view pattern = "^((((((((((a+))))))))))$";
    struct LimitCase
    {
        rad::JsonSchemaRegexMatchLimits limits;
        std::string_view message;
    };
    constexpr LimitCase limits[] = {
        {{0, 1'000, 8'192}, "match limit"},     {{1, 1'000, 8'192}, "match limit"},
        {{1'000'000, 0, 8'192}, "depth limit"}, {{1'000'000, 1, 8'192}, "depth limit"},
        {{1'000'000, 1'000, 0}, "heap limit"},  {{1'000'000, 1'000, 1}, "heap limit"},
    };
    struct SchemaCase
    {
        rad::JsonValue schema;
        rad::JsonValue instance;
        std::string_view path;
    };
    const SchemaCase schemas[] = {
        {rad::JsonObject{{"pattern", pattern}}, "aaaa", "/pattern"},
        {rad::JsonObject{{"patternProperties", rad::JsonObject{{pattern, true}}}},
         rad::JsonObject{{"aaaa", 1}}, "/patternProperties/^((((((((((a+))))))))))$"},
        {rad::JsonObject{{"anyOf", rad::JsonArray{rad::JsonObject{{"pattern", pattern}}, true}}},
         "aaaa", "/anyOf/0/pattern"},
    };
    rad::JsonSchemaCompileOptions compileOptions;
    compileOptions.retrievalUri = "https://example.com/root.json";
    for (const auto& schema : schemas)
    {
        const auto compiled = rad::JsonSchema::Compile(
            schema.schema, rad::JsonSchemaDialect::Draft2020_12, compileOptions);
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(schema.instance));
        for (const auto& limit : limits)
        {
            SCOPED_TRACE(limit.message);
            SCOPED_TRACE(std::format("match={}, backtrackingDepth={}, heapKiB={}",
                                     limit.limits.matchLimit, limit.limits.backtrackingDepthLimit,
                                     limit.limits.heapLimitKiB));
            rad::JsonSchemaValidationOptions options;
            options.regex = limit.limits;
            options.maxErrors = 1;
            const auto result = compiled.value().Validate(schema.instance, options);
            ASSERT_EQ(result.errors.size(), 1);
            EXPECT_NE(result.errors[0].message.find(limit.message), std::string::npos);
            EXPECT_EQ(result.errors[0].schemaPath, schema.path);
            EXPECT_EQ(result.errors[0].schemaUri, compileOptions.retrievalUri);
        }
        EXPECT_TRUE(compiled.value().Validate(schema.instance));
    }

    const auto compiled =
        rad::JsonSchema::Compile(schemas[0].schema, rad::JsonSchemaDialect::Draft2020_12);
    ASSERT_TRUE(compiled);
    std::vector<std::future<bool>> workers;
    for (std::size_t worker = 0; worker < 4; ++worker)
    {
        workers.push_back(
            std::async(std::launch::async,
                       [schema = compiled.value(), worker]
                       {
                           rad::JsonSchemaValidationOptions options;
                           const bool expected = worker % 2 == 0;
                           options.regex.matchLimit = expected ? 2'000'000 : 0;
                           for (std::size_t iteration = 0; iteration < 16; ++iteration)
                           {
                               if (static_cast<bool>(schema.Validate("aaaa", options)) != expected)
                               {
                                   return false;
                               }
                           }
                           return true;
                       }));
    }
    for (auto& worker : workers)
    {
        EXPECT_TRUE(worker.get());
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

TEST(IO, JsonSchemaDocumentRegistry)
{
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const auto makeSchema = [&]
        {
            rad::JsonSchemaCompileOptions options;
            options.retrievalUri = "https://example.com/root.json";
            options.documents = {
                {"https://example.com/a.json",
                 rad::ParseJson(R"json({"$id": "canonical/a.json", "type": "object",
                     "properties": {"child": {"$ref": "../b.json#/properties/value"}}})json")
                     .value()},
                {"https://example.com/b.json",
                 rad::ParseJson(R"json({"properties": {"value": {"type": "integer"},
                     "embedded": {"$id": "embedded.json", "type": "string"}}})json")
                     .value()},
            };
            auto root = rad::JsonObject{{"$ref", "canonical/a.json"}};
            if (dialect != rad::JsonSchemaDialect::Draft7)
            {
                root["unevaluatedProperties"] = false;
            }
            return rad::JsonSchema::Compile(root, dialect, options);
        };
        const auto compiled = makeSchema();
        ASSERT_TRUE(compiled) << compiled.error().schemaUri << compiled.error().schemaPath << ": "
                              << compiled.error().message;
        auto copied = compiled.value();
        auto moved = std::move(copied);
        EXPECT_TRUE(moved.Validate(rad::JsonObject{{"child", 1}}));
        rad::JsonSchemaValidationOptions diagnosticLimit;
        diagnosticLimit.maxErrors = 1;
        const auto invalid = moved.Validate(rad::JsonObject{{"child", "invalid"}}, diagnosticLimit);
        ASSERT_FALSE(invalid);
        ASSERT_EQ(invalid.errors.size(), 1);
        EXPECT_EQ(invalid.errors[0].instancePath, "/child");
        EXPECT_EQ(invalid.errors[0].schemaPath, "/properties/value/type");
        EXPECT_EQ(invalid.errors[0].schemaUri, "https://example.com/b.json");
        if (dialect != rad::JsonSchemaDialect::Draft7)
        {
            EXPECT_FALSE(moved.Validate(rad::JsonObject{{"child", 1}, {"extra", 1}}));
        }

        rad::JsonSchemaCompileOptions options;
        options.retrievalUri = "https://example.com/root.json";
        options.documents = {
            {"https://example.com/container.json", rad::ParseJson(R"json({"properties":
                 {"embedded": {"$id": "embedded.json", "type": "integer"}}})json")
                                                       .value()},
        };
        const auto embedded =
            rad::JsonSchema::Compile(rad::JsonObject{{"$ref", "embedded.json"}}, dialect, options);
        ASSERT_TRUE(embedded) << embedded.error().message;
        EXPECT_TRUE(embedded.value().Validate(1));
        const auto failure = embedded.value().Validate("invalid");
        ASSERT_FALSE(failure);
        EXPECT_EQ(failure.errors[0].schemaUri, "https://example.com/container.json");
        EXPECT_EQ(failure.errors[0].schemaPath, "/properties/embedded/type");

        options.documents = {
            {"https://example.com/alias.json", rad::ParseJson(R"json({"$id": "canonical.json",
                                   "properties": {"value": {"$anchor": "value",
                                                           "type": "integer"}}})json")
                                                   .value()},
        };
        if (dialect == rad::JsonSchemaDialect::Draft7)
        {
            auto& value = options.documents[0]
                              .schema.as_object()
                              .at("properties")
                              .as_object()
                              .at("value")
                              .as_object();
            value.erase("$anchor");
            value["$id"] = "#value";
        }
        for (const auto* reference :
             {"alias.json#value", "canonical.json#value", "alias.json#/properties/value"})
        {
            const auto aliased =
                rad::JsonSchema::Compile(rad::JsonObject{{"$ref", reference}}, dialect, options);
            ASSERT_TRUE(aliased) << aliased.error().message;
            const auto invalid = aliased.value().Validate("invalid");
            ASSERT_FALSE(invalid);
            EXPECT_EQ(invalid.errors[0].schemaUri, "https://example.com/alias.json");
            EXPECT_EQ(invalid.errors[0].schemaPath, "/properties/value/type");
        }

        options.documents = {
            {"https://example.com/a.json", rad::JsonObject{{"$ref", "b.json"}}},
            {"https://example.com/b.json", rad::JsonObject{{"$ref", "a.json"}}},
        };
        const auto cyclic =
            rad::JsonSchema::Compile(rad::JsonObject{{"$ref", "a.json"}}, dialect, options);
        ASSERT_TRUE(cyclic) << cyclic.error().message;
        rad::JsonSchemaValidationOptions limits;
        limits.maxDepth = 2;
        limits.maxErrors = 1;
        const auto exhausted = cyclic.value().Validate(1, limits);
        ASSERT_FALSE(exhausted);
        ASSERT_EQ(exhausted.errors.size(), 1);
        EXPECT_EQ(exhausted.errors[0].schemaPath, "");
        EXPECT_EQ(exhausted.errors[0].schemaUri, "https://example.com/a.json");
        EXPECT_EQ(exhausted.errors[0].message, "maximum validation depth exceeded");

        options.documents = {
            {"https://example.com/a.json",
             rad::ParseJson(R"json({"type": "integer",
                 "properties": {"child": {"allOf": [true]}}})json").value()},
        };
        const auto branches = rad::ParseJson(R"json({"anyOf": [{"$ref": "a.json"}, true]})json");
        ASSERT_TRUE(branches);
        const auto limited = rad::JsonSchema::Compile(branches.value(), dialect, options);
        ASSERT_TRUE(limited) << limited.error().message;
        limits.maxDepth = 3;
        for (const std::size_t maxErrors : {0U, 1U, 2U})
        {
            limits.maxErrors = maxErrors;
            const auto result =
                limited.value().Validate(rad::JsonObject{{"child", rad::JsonObject{}}}, limits);
            ASSERT_FALSE(result);
            ASSERT_EQ(result.errors.size(), 1);
            EXPECT_EQ(result.errors[0].schemaUri, "https://example.com/a.json");
            EXPECT_EQ(result.errors[0].schemaPath, "/properties/child/allOf/0");
            EXPECT_EQ(result.errors[0].message, "maximum validation depth exceeded");
        }
        limits.maxDepth = 4;
        EXPECT_TRUE(
            limited.value().Validate(rad::JsonObject{{"child", rad::JsonObject{}}}, limits));
    }
}

TEST(IO, JsonSchemaDocumentRegistryDiagnostics)
{
    constexpr auto dialect = rad::JsonSchemaDialect::Draft2020_12;
    rad::JsonSchemaCompileOptions options;
    options.retrievalUri = "https://example.com/root.json";
    const auto root = rad::JsonObject{{"$ref", "child.json"}};
    const auto missing = rad::JsonSchema::Compile(root, dialect, options);
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    EXPECT_EQ(missing.error().schemaPath, "/$ref");
    EXPECT_EQ(missing.error().schemaUri, options.retrievalUri);

    options.documents = {
        {"https://example.com/child.json", rad::JsonObject{{"type", 42}}},
    };
    const auto invalid = rad::JsonSchema::Compile(root, dialect, options);
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.error().schemaPath, "/type");
    EXPECT_EQ(invalid.error().schemaUri, "https://example.com/child.json");
    EXPECT_TRUE(rad::JsonSchema::Compile(rad::JsonObject{}, dialect, options));

    options.documents[0].schema = rad::JsonObject{
        {"$schema", "http://json-schema.org/draft-07/schema"},
        {"properties", rad::JsonObject{{"child", true}}},
    };
    const auto mixed = rad::JsonSchema::Compile(
        rad::JsonObject{{"$ref", "child.json#/properties/child"}}, dialect, options);
    ASSERT_FALSE(mixed);
    EXPECT_EQ(mixed.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    EXPECT_EQ(mixed.error().schemaPath, "/$schema");
    EXPECT_EQ(mixed.error().schemaUri, "https://example.com/child.json");

    options.documents[0].schema = true;
    options.documents.push_back({"https://EXAMPLE.COM/%63hild.json", false});
    const auto duplicate = rad::JsonSchema::Compile(root, dialect, options);
    ASSERT_FALSE(duplicate);
    EXPECT_EQ(duplicate.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    options.documents.pop_back();
    const auto conflict = rad::JsonSchema::Compile(
        rad::JsonObject{{"$id", "https://example.com/child.json"}}, dialect, options);
    ASSERT_FALSE(conflict);
    EXPECT_EQ(conflict.error().schemaPath, "/$id");
    EXPECT_EQ(conflict.error().schemaUri, options.retrievalUri);

    auto detected = rad::JsonObject{{"$schema", "https://json-schema.org/draft/2020-12/schema"},
                                    {"$ref", "child.json"}};
    EXPECT_TRUE(rad::JsonSchema::Compile(detected, options));
    EXPECT_FALSE(rad::JsonSchema::Compile(detected));
    options.retrievalUri = "relative.json";
    const auto relative = rad::JsonSchema::Compile(root, dialect, options);
    ASSERT_FALSE(relative);
    EXPECT_EQ(relative.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    options.retrievalUri = "https://example.com/root.json#fragment";
    EXPECT_FALSE(rad::JsonSchema::Compile(root, dialect, options));

    options.retrievalUri = "https://example.com/root.json";
    options.documents[0].schema = rad::JsonObject{{"$id", 42}};
    const auto ignored = rad::ParseJson(
        R"json({"$ref": "#/definitions/value",
                 "definitions": {"value": true, "unused": {"$ref": "child.json"}}})json");
    ASSERT_TRUE(ignored);
    EXPECT_TRUE(rad::JsonSchema::Compile(ignored.value(), rad::JsonSchemaDialect::Draft7, options));
    const auto malformedId = rad::JsonSchema::Compile(root, dialect, options);
    ASSERT_FALSE(malformedId);
    EXPECT_EQ(malformedId.error().schemaPath, "/$id");
    EXPECT_EQ(malformedId.error().schemaUri, "https://example.com/child.json");
}

TEST(IO, JsonSchemaRecursiveReferenceDiagnostics)
{
    constexpr auto dialect = rad::JsonSchemaDialect::Draft2019_09;
    struct Case
    {
        std::string_view schema;
        std::string_view path;
        rad::JsonSchemaCompileErrorCode code;
    };
    constexpr Case cases[] = {
        {R"json({"$recursiveAnchor": 1})json", "/$recursiveAnchor",
         rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json({"$recursiveRef": false})json", "/$recursiveRef",
         rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json({"$recursiveRef": "#/%ZZ"})json", "/$recursiveRef",
         rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json({"$recursiveRef": "#/$defs/value", "$defs": {"value": true}})json",
         "/$recursiveRef", rad::JsonSchemaCompileErrorCode::UnsupportedFeature},
        {R"json({"$recursiveRef": "other.json#"})json", "/$recursiveRef",
         rad::JsonSchemaCompileErrorCode::UnsupportedFeature},
        {R"json({"properties": {"child": {"$recursiveAnchor": "true"}}})json",
         "/properties/child/$recursiveAnchor", rad::JsonSchemaCompileErrorCode::InvalidSchema},
    };
    rad::JsonSchemaCompileOptions options;
    options.retrievalUri = "https://example.com/root.json";
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto parsed = rad::ParseJson(testCase.schema);
        ASSERT_TRUE(parsed);
        const auto compiled = rad::JsonSchema::Compile(parsed.value(), dialect, options);
        ASSERT_FALSE(compiled);
        EXPECT_EQ(compiled.error().code, testCase.code);
        EXPECT_EQ(compiled.error().schemaPath, testCase.path);
        EXPECT_EQ(compiled.error().schemaUri, options.retrievalUri);
    }
    for (const auto otherDraft :
         {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2020_12})
    {
        const auto ignored = rad::JsonSchema::Compile(
            rad::JsonObject{{"$recursiveRef", false}, {"$recursiveAnchor", 1}}, otherDraft);
        ASSERT_TRUE(ignored) << ignored.error().message;
        EXPECT_TRUE(ignored.value().Validate(1));
    }
}

TEST(IO, JsonSchemaRecursiveReferenceRegistryAndCopies)
{
    const auto makeSchema = []
    {
        rad::JsonSchemaCompileOptions options;
        options.retrievalUri = "https://example.com/strict.json";
        options.documents = {
            {"https://example.com/tree.json",
             rad::ParseJson(R"json({"$id": "trees/base.json", "$recursiveAnchor": true,
                 "type": ["object", "integer"],
                 "properties": {"child": {"$recursiveRef": "#"},
                                "static": {"$ref": "#"}}})json").value()},
        };
        return rad::JsonSchema::Compile(
            rad::JsonObject{{"$recursiveAnchor", true}, {"$ref", "tree.json"}, {"minimum", 2}},
            rad::JsonSchemaDialect::Draft2019_09, options);
    };
    const auto compiled = makeSchema();
    ASSERT_TRUE(compiled) << compiled.error().message;
    auto copied = compiled.value();
    auto moved = std::move(copied);
    for (const std::size_t maxErrors : {0U, 1U, 2U})
    {
        rad::JsonSchemaValidationOptions options;
        options.maxErrors = maxErrors;
        EXPECT_TRUE(moved.Validate(rad::JsonObject{{"child", 2}, {"static", 1}}, options));
        const auto invalid = moved.Validate(rad::JsonObject{{"child", 1}}, options);
        ASSERT_FALSE(invalid);
        ASSERT_EQ(invalid.errors.size(), 1);
        EXPECT_EQ(invalid.errors[0].instancePath, "/child");
        EXPECT_EQ(invalid.errors[0].schemaPath, "/minimum");
        EXPECT_EQ(invalid.errors[0].schemaUri, "https://example.com/strict.json");
        EXPECT_TRUE(moved.Validate(rad::JsonObject{{"child", 2}}, options));
    }
}

TEST(IO, JsonSchemaRecursiveReferenceResourceLimits)
{
    struct Case
    {
        std::string_view schema;
        std::string_view data;
        std::size_t maxDepth;
    };
    constexpr Case cases[] = {
        {R"json({"$recursiveAnchor": true, "$recursiveRef": "#"})json", "1", 4},
        {R"json({"$recursiveAnchor": true, "type": "integer",
                 "properties": {"child": {"$recursiveRef": "#"}}})json",
         R"json({"child": {}})json", 3},
    };
    rad::JsonSchemaCompileOptions compileOptions;
    compileOptions.retrievalUri = "https://example.com/root.json";
    const auto schema = rad::ParseJson(R"json({"anyOf": [{"$ref": "cycle.json"}, true]})json");
    ASSERT_TRUE(schema);
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto document = rad::ParseJson(testCase.schema);
        const auto data = rad::ParseJson(testCase.data);
        ASSERT_TRUE(document);
        ASSERT_TRUE(data);
        compileOptions.documents = {{"https://example.com/cycle.json", document.value()}};
        const auto compiled = rad::JsonSchema::Compile(
            schema.value(), rad::JsonSchemaDialect::Draft2019_09, compileOptions);
        ASSERT_TRUE(compiled) << compiled.error().message;
        for (const std::size_t maxErrors : {0U, 1U, 2U})
        {
            rad::JsonSchemaValidationOptions options;
            options.maxErrors = maxErrors;
            options.maxDepth = testCase.maxDepth;
            const auto result = compiled.value().Validate(data.value(), options);
            ASSERT_FALSE(result);
            ASSERT_EQ(result.errors.size(), 1);
            EXPECT_EQ(result.errors[0].instancePath, testCase.maxDepth == 3 ? "/child" : "");
            EXPECT_EQ(result.errors[0].schemaPath, "");
            EXPECT_EQ(result.errors[0].schemaUri, "https://example.com/cycle.json");
            EXPECT_EQ(result.errors[0].message, "maximum validation depth exceeded");
        }
    }
}

TEST(IO, JsonSchemaDynamicReferenceDiagnostics)
{
    constexpr auto dialect = rad::JsonSchemaDialect::Draft2020_12;
    struct Case
    {
        std::string_view schema;
        std::string_view path;
        rad::JsonSchemaCompileErrorCode code = rad::JsonSchemaCompileErrorCode::InvalidSchema;
    };
    constexpr Case cases[] = {
        {R"json({"$dynamicAnchor": false})json", "/$dynamicAnchor"},
        {R"json({"$dynamicAnchor": ""})json", "/$dynamicAnchor"},
        {R"json({"$dynamicAnchor": "1node"})json", "/$dynamicAnchor"},
        {R"json({"$dynamicAnchor": "node:child"})json", "/$dynamicAnchor"},
        {R"json({"$dynamicAnchor": "node/child"})json", "/$dynamicAnchor"},
        {R"json({"$dynamicRef": 42})json", "/$dynamicRef"},
        {R"json({"$dynamicRef": "#/%ZZ"})json", "/$dynamicRef"},
        {R"json({"$dynamicRef": "#unknown"})json", "/$dynamicRef"},
        {R"json({"$dynamicRef": "#/missing"})json", "/$dynamicRef"},
        {R"json({"properties": {"value": {"$dynamicAnchor": 1}}})json",
         "/properties/value/$dynamicAnchor"},
        {R"json({"$defs": {"one": {"$anchor": "node"},
                          "two": {"$dynamicAnchor": "node"}}})json",
         "/$defs/two/$dynamicAnchor"},
        {R"json({"$defs": {"one": {"$dynamicAnchor": "node"},
                          "two": {"$dynamicAnchor": "node"}}})json",
         "/$defs/two/$dynamicAnchor"},
        {R"json({"$dynamicRef": "external.json#node"})json", "/$dynamicRef",
         rad::JsonSchemaCompileErrorCode::UnsupportedFeature},
    };
    rad::JsonSchemaCompileOptions options;
    options.retrievalUri = "https://example.com/root.json";
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto parsed = rad::ParseJson(testCase.schema);
        ASSERT_TRUE(parsed);
        const auto compiled = rad::JsonSchema::Compile(parsed.value(), dialect, options);
        ASSERT_FALSE(compiled);
        EXPECT_EQ(compiled.error().code, testCase.code);
        EXPECT_EQ(compiled.error().schemaPath, testCase.path);
        EXPECT_EQ(compiled.error().schemaUri, options.retrievalUri);
    }
    EXPECT_TRUE(rad::JsonSchema::Compile(
        rad::JsonObject{{"$dynamicAnchor", "_node-1.2"}}, dialect));
    for (const auto otherDraft :
         {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09})
    {
        const auto ignored = rad::JsonSchema::Compile(
            rad::JsonObject{{"$dynamicRef", false}, {"$dynamicAnchor", 1}}, otherDraft);
        ASSERT_TRUE(ignored) << ignored.error().message;
        EXPECT_TRUE(ignored.value().Validate(1));
    }
}

TEST(IO, JsonSchemaDynamicReferenceRegistryAndCopies)
{
    const auto makeSchema = [](std::string_view reference)
    {
        rad::JsonSchemaCompileOptions options;
        options.retrievalUri = "https://example.com/strict.json";
        options.documents = {
            {"https://example.com/tree.json",
             rad::ParseJson(R"json({"$id": "trees/base.json", "$dynamicAnchor": "node",
                 "type": ["object", "integer"],
                 "properties": {"child": {"$dynamicRef": "#node"},
                                "static": {"$ref": "#node"}}})json").value()},
        };
        return rad::JsonSchema::Compile(
            rad::JsonObject{{"$dynamicAnchor", "node"}, {"$ref", reference}, {"minimum", 2}},
            rad::JsonSchemaDialect::Draft2020_12, options);
    };
    for (const std::string_view reference :
         {"tree.json#node", "trees/base.json#node", "tree.json#%6Eode"})
    {
        SCOPED_TRACE(reference);
        const auto compiled = makeSchema(reference);
        ASSERT_TRUE(compiled) << compiled.error().message;
        auto copied = compiled.value();
        auto moved = std::move(copied);
        for (const std::size_t maxErrors : {0U, 1U, 2U})
        {
            rad::JsonSchemaValidationOptions options;
            options.maxErrors = maxErrors;
            EXPECT_TRUE(moved.Validate(rad::JsonObject{{"child", 2}, {"static", 1}}, options));
            const auto invalid = moved.Validate(rad::JsonObject{{"child", 1}}, options);
            ASSERT_FALSE(invalid);
            ASSERT_EQ(invalid.errors.size(), 1);
            EXPECT_EQ(invalid.errors[0].instancePath, "/child");
            EXPECT_EQ(invalid.errors[0].schemaPath, "/minimum");
            EXPECT_EQ(invalid.errors[0].schemaUri, "https://example.com/strict.json");
            EXPECT_TRUE(moved.Validate(rad::JsonObject{{"child", 2}}, options));
        }
    }
}

TEST(IO, JsonSchemaDynamicAnchorTargetCompilation)
{
    constexpr auto dialect = rad::JsonSchemaDialect::Draft2020_12;
    rad::JsonSchemaCompileOptions options;
    options.retrievalUri = "https://example.com/root.json";
    const auto document = rad::ParseJson(R"json({"$defs": {
        "entry": {"$dynamicRef": "other.json#node"},
        "hidden": {"$dynamicAnchor": "node", "type": 42}
    }})json");
    ASSERT_TRUE(document);
    options.documents = {
        {"https://example.com/child.json", document.value()},
        {"https://example.com/other.json",
         rad::JsonObject{{"$dynamicAnchor", "node"}, {"type", "string"}}},
    };
    const auto root = rad::JsonObject{{"$ref", "child.json#/$defs/entry"}};
    const auto invalid = rad::JsonSchema::Compile(root, dialect, options);
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    EXPECT_EQ(invalid.error().schemaPath, "/$defs/hidden/type");
    EXPECT_EQ(invalid.error().schemaUri, "https://example.com/child.json");
    EXPECT_TRUE(rad::JsonSchema::Compile(rad::JsonObject{}, dialect, options));

    options.documents[0].schema.as_object().at("$defs").as_object()
        .at("hidden").as_object()["type"] = "integer";
    const auto compiled = rad::JsonSchema::Compile(root, dialect, options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    EXPECT_TRUE(compiled.value().Validate(1));
    const auto result = compiled.value().Validate("invalid");
    ASSERT_FALSE(result);
    ASSERT_EQ(result.errors.size(), 1);
    EXPECT_EQ(result.errors[0].schemaPath, "/$defs/hidden/type");
    EXPECT_EQ(result.errors[0].schemaUri, "https://example.com/child.json");
}

TEST(IO, JsonSchemaDynamicReferenceResourceLimits)
{
    struct Case
    {
        std::string_view schema;
        std::string_view data;
        std::size_t maxDepth;
        std::string_view instancePath;
    };
    constexpr Case cases[] = {
        {R"json({"$dynamicAnchor": "node", "$dynamicRef": "#node"})json", "1", 4, ""},
        {R"json({"$dynamicAnchor": "node", "type": "integer",
                 "properties": {"child": {"$dynamicRef": "#node"}}})json",
         R"json({"child": {}})json", 3, "/child"},
    };
    rad::JsonSchemaCompileOptions compileOptions;
    compileOptions.retrievalUri = "https://example.com/root.json";
    const auto schema = rad::ParseJson(R"json({"anyOf": [{"$ref": "cycle.json"}, true]})json");
    ASSERT_TRUE(schema);
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto document = rad::ParseJson(testCase.schema);
        const auto data = rad::ParseJson(testCase.data);
        ASSERT_TRUE(document);
        ASSERT_TRUE(data);
        compileOptions.documents = {{"https://example.com/cycle.json", document.value()}};
        const auto compiled = rad::JsonSchema::Compile(
            schema.value(), rad::JsonSchemaDialect::Draft2020_12, compileOptions);
        ASSERT_TRUE(compiled) << compiled.error().message;
        for (const std::size_t maxErrors : {0U, 1U, 2U})
        {
            rad::JsonSchemaValidationOptions options;
            options.maxErrors = maxErrors;
            options.maxDepth = testCase.maxDepth;
            const auto result = compiled.value().Validate(data.value(), options);
            ASSERT_FALSE(result);
            ASSERT_EQ(result.errors.size(), 1);
            EXPECT_EQ(result.errors[0].instancePath, testCase.instancePath);
            EXPECT_EQ(result.errors[0].schemaPath, "");
            EXPECT_EQ(result.errors[0].schemaUri, "https://example.com/cycle.json");
            EXPECT_EQ(result.errors[0].message, "maximum validation depth exceeded");
        }
    }
}

TEST(IO, JsonSchemaNonFiniteInstances)
{
    const rad::JsonValue schemas[] = {
        true,
        false,
        rad::JsonObject{},
        rad::JsonObject{{"const", std::int64_t{1}}},
        rad::JsonObject{{"enum", rad::JsonArray{std::numeric_limits<std::uint64_t>::max(), 2.0}}},
        rad::JsonObject{{"anyOf", rad::JsonArray{true, rad::JsonObject{{"minimum", 0}}}}},
    };
    rad::JsonSchemaCompileOptions compileOptions;
    compileOptions.retrievalUri = "https://example.com/root.json";
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        for (const auto& schema : schemas)
        {
            const auto compiled = rad::JsonSchema::Compile(schema, dialect, compileOptions);
            ASSERT_TRUE(compiled) << compiled.error().message;
            for (const double value :
                 {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()})
            {
                const auto result = compiled.value().Validate(value);
                ASSERT_EQ(result.errors.size(), 1);
                EXPECT_EQ(result.errors[0].message, "number must be finite");
                EXPECT_EQ(result.errors[0].instancePath, "");
                EXPECT_EQ(result.errors[0].schemaPath, "");
                EXPECT_EQ(result.errors[0].schemaUri, compileOptions.retrievalUri);
            }
        }
        const auto permissive = rad::JsonSchema::Compile(true, dialect, compileOptions);
        ASSERT_TRUE(permissive);
        const rad::JsonValue nested =
            rad::JsonObject{{"x/~", rad::JsonArray{1, std::numeric_limits<double>::quiet_NaN()}},
                            {"other", std::numeric_limits<double>::infinity()}};
        for (const std::size_t maxErrors : {0U, 1U, 2U, 64U})
        {
            rad::JsonSchemaValidationOptions options;
            options.maxErrors = maxErrors;
            options.maxDepth = 0;
            const auto result = permissive.value().Validate(nested, options);
            ASSERT_EQ(result.errors.size(), maxErrors <= 1 ? 1 : 2);
            EXPECT_EQ(result.errors[0].instancePath, "/x~1~0/1");
            EXPECT_EQ(result.errors[0].schemaPath, "");
            if (result.errors.size() == 2)
            {
                EXPECT_EQ(result.errors[1].instancePath, "/other");
            }
        }
        rad::JsonValue deep = 1;
        std::string deepPath;
        for (std::size_t depth = 0; depth < 256; ++depth)
        {
            deep = rad::JsonArray{std::move(deep)};
            deepPath += "/0";
        }
        rad::JsonSchemaValidationOptions options;
        options.maxDepth = 0;
        EXPECT_TRUE(permissive.value().Validate(deep, options));
        auto* leaf = &deep;
        for (std::size_t depth = 0; depth < 256; ++depth)
        {
            leaf = &leaf->as_array()[0];
        }
        *leaf = std::numeric_limits<double>::quiet_NaN();
        const auto deepInvalid = permissive.value().Validate(deep, options);
        ASSERT_EQ(deepInvalid.errors.size(), 1);
        EXPECT_EQ(deepInvalid.errors[0].instancePath, deepPath);
        EXPECT_EQ(deepInvalid.errors[0].message, "number must be finite");
    }
}

TEST(IO, JsonSchemaNonFiniteLiterals)
{
    rad::JsonSchemaCompileOptions options;
    options.retrievalUri = "https://example.com/root.json";
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        for (const double value :
             {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity()})
        {
            struct Case
            {
                rad::JsonValue schema;
                std::string_view path;
            };
            const Case cases[] = {
                {rad::JsonObject{{"const", value}}, "/const"},
                {rad::JsonObject{{"enum", rad::JsonArray{1, value, value}}}, "/enum/1"},
                {rad::JsonObject{{"const", rad::JsonObject{{"/~", rad::JsonArray{value}}}}},
                 "/const/~1~0/0"},
                {rad::JsonObject{
                     {"enum", rad::JsonArray{1, rad::JsonObject{{"/~", rad::JsonArray{value}}}}}},
                 "/enum/1/~1~0/0"},
            };
            for (const auto& testCase : cases)
            {
                const auto compiled = rad::JsonSchema::Compile(testCase.schema, dialect, options);
                ASSERT_FALSE(compiled);
                EXPECT_EQ(compiled.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
                EXPECT_EQ(compiled.error().schemaPath, testCase.path);
                EXPECT_EQ(compiled.error().schemaUri, options.retrievalUri);
            }
        }
        const auto nan = std::numeric_limits<double>::quiet_NaN();
        for (const rad::JsonArray names :
             {rad::JsonArray{std::int64_t{1}, nan}, rad::JsonArray{std::uint64_t{1}, nan},
              rad::JsonArray{nan, std::int64_t{1}}, rad::JsonArray{nan, std::uint64_t{1}},
              rad::JsonArray{nan, nan}})
        {
            const auto invalidNames =
                rad::JsonSchema::Compile(rad::JsonObject{{"required", names}}, dialect, options);
            ASSERT_FALSE(invalidNames);
            EXPECT_EQ(invalidNames.error().schemaPath, "/required/0");
        }
        options.documents = {
            {"https://example.com/unused.json",
             rad::JsonObject{{"const", std::numeric_limits<double>::quiet_NaN()}}},
        };
        EXPECT_TRUE(rad::JsonSchema::Compile(true, dialect, options));
        const auto referenced =
            rad::JsonSchema::Compile(rad::JsonObject{{"$ref", "unused.json"}}, dialect, options);
        ASSERT_FALSE(referenced);
        EXPECT_EQ(referenced.error().schemaPath, "/const");
        EXPECT_EQ(referenced.error().schemaUri, options.documents[0].uri);
    }
    options.documents = {
        {"https://example.com/meta",
         rad::JsonObject{
             {"$schema", "https://json-schema.org/draft/2020-12/schema"},
             {"$vocabulary",
              rad::JsonObject{{"https://json-schema.org/draft/2020-12/vocab/core", true}}}}},
    };
    const auto disabled = rad::JsonSchema::Compile(
        rad::JsonObject{{"$schema", "https://example.com/meta"},
                        {"const", std::numeric_limits<double>::quiet_NaN()},
                        {"enum", rad::JsonArray{std::numeric_limits<double>::infinity()}}},
        options);
    ASSERT_TRUE(disabled) << disabled.error().message;
    EXPECT_TRUE(disabled.value().Validate(1));
    EXPECT_FALSE(disabled.value().Validate(std::numeric_limits<double>::quiet_NaN()));
}

TEST(IO, JsonSchemaNumericComparisonBoundaries)
{
    struct Case
    {
        rad::JsonValue value;
        rad::JsonValue limit;
        int ordering;
    };
    const Case cases[] = {
        {std::int64_t{1}, std::uint64_t{1}, 0},
        {std::int64_t{-1}, std::numeric_limits<std::uint64_t>::max(), -1},
        {std::numeric_limits<std::uint64_t>::max(), std::int64_t{-1}, 1},
        {std::numeric_limits<std::int64_t>::min(), -9223372036854775808.0, 0},
        {std::numeric_limits<std::int64_t>::max(), 9223372036854775808.0, -1},
        {9223372036854775808.0, std::numeric_limits<std::int64_t>::max(), 1},
        {std::numeric_limits<std::uint64_t>::max(), 18446744073709551616.0, -1},
        {18446744073709551616.0, std::numeric_limits<std::uint64_t>::max(), 1},
        {std::nextafter(18446744073709551616.0, 0.0), std::uint64_t{18446744073709549568ULL}, 0},
        {-1.5, std::int64_t{-1}, -1},
        {1.5, std::uint64_t{1}, 1},
        {-0.0, std::uint64_t{0}, 0},
    };
    for (const auto& testCase : cases)
    {
        for (const auto keyword : {"const", "minimum", "maximum"})
        {
            const auto compiled = rad::JsonSchema::Compile(
                rad::JsonObject{{keyword, testCase.limit}}, rad::JsonSchemaDialect::Draft2020_12);
            ASSERT_TRUE(compiled);
            const bool expected = std::string_view(keyword) == "const"     ? testCase.ordering == 0
                                  : std::string_view(keyword) == "minimum" ? testCase.ordering >= 0
                                                                           : testCase.ordering <= 0;
            EXPECT_EQ(static_cast<bool>(compiled.value().Validate(testCase.value)), expected);
        }
    }
}

TEST(IO, JsonSchemaMultipleOfDecimalSemantics)
{
    struct Case
    {
        rad::JsonValue value;
        rad::JsonValue divisor;
        bool valid;
    };
    const Case cases[] = {
        {0.3, 0.1, true},
        {0.1 + 0.2, 0.1, false},
        {std::nextafter(0.3, 0.0), 0.1, false},
        {std::nextafter(0.3, 1.0), 0.1, false},
        {0.3, std::nextafter(0.1, 0.0), false},
        {0.3, std::nextafter(0.1, 1.0), false},
        {-0.0, std::numeric_limits<double>::denorm_min(), true},
        {std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::denorm_min(),
         true},
        {std::numeric_limits<double>::denorm_min(), 1e-323, false},
        {1e-323, std::numeric_limits<double>::denorm_min(), true},
        {std::numeric_limits<double>::max(), 0.5, true},
        {std::numeric_limits<double>::max(), 3, false},
        {std::numeric_limits<std::uint64_t>::max(), 0.5, true},
        {std::numeric_limits<std::uint64_t>::max(), 10.0, false},
        {std::numeric_limits<std::int64_t>::min(), 0.5, true},
        {std::numeric_limits<std::int64_t>::min(), 3, false},
        {0.125, 0.25, false},
        {0.25, 0.125, true},
        {1e-300, 1e300, false},
        {1e300, 1e-300, true},
    };
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        for (const auto& testCase : cases)
        {
            SCOPED_TRACE(rad::PrettyJson(testCase.value));
            SCOPED_TRACE(rad::PrettyJson(testCase.divisor));
            const auto compiled = rad::JsonSchema::Compile(
                rad::JsonObject{{"multipleOf", testCase.divisor}}, dialect);
            ASSERT_TRUE(compiled) << compiled.error().message;
            rad::JsonSchemaValidationOptions options;
            options.maxErrors = 1;
            const auto result = compiled.value().Validate(testCase.value, options);
            EXPECT_EQ(static_cast<bool>(result), testCase.valid) << FormatValidationErrors(result);
            if (!testCase.valid)
            {
                ASSERT_EQ(result.errors.size(), 1);
                EXPECT_EQ(result.errors[0].schemaPath, "/multipleOf");
            }
        }
        for (const double divisor : {0.0, -0.1, std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN()})
        {
            const auto compiled =
                rad::JsonSchema::Compile(rad::JsonObject{{"multipleOf", divisor}}, dialect);
            ASSERT_FALSE(compiled);
            EXPECT_EQ(compiled.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
            EXPECT_EQ(compiled.error().schemaPath, "/multipleOf");
        }
    }
}

TEST(IO, JsonSchemaMultipleOfDecimalGrid)
{
    const auto powerOfTen = [](int exponent)
    {
        std::int64_t power = 1;
        for (int index = 0; index < exponent; ++index)
        {
            power *= 10;
        }
        return power;
    };
    for (int divisorCoefficient = 1; divisorCoefficient <= 25; ++divisorCoefficient)
    {
        for (int divisorExponent = -3; divisorExponent <= 3; ++divisorExponent)
        {
            const auto divisor =
                rad::ParseJson(std::format("{}e{}", divisorCoefficient, divisorExponent));
            ASSERT_TRUE(divisor);
            const auto compiled =
                rad::JsonSchema::Compile(rad::JsonObject{{"multipleOf", divisor.value()}},
                                         rad::JsonSchemaDialect::Draft2020_12);
            ASSERT_TRUE(compiled) << compiled.error().message;
            for (int coefficient = -50; coefficient <= 50; ++coefficient)
            {
                for (int exponent = -3; exponent <= 3; ++exponent)
                {
                    const auto text = std::format("{}e{}", coefficient, exponent);
                    const auto instance = rad::ParseJson(text);
                    ASSERT_TRUE(instance);
                    const int difference = exponent - divisorExponent;
                    const auto numerator = coefficient * powerOfTen(std::max(difference, 0));
                    const auto denominator =
                        divisorCoefficient * powerOfTen(std::max(-difference, 0));
                    const bool expected = numerator % denominator == 0;
                    const auto result = compiled.value().Validate(instance.value());
                    EXPECT_EQ(static_cast<bool>(result), expected)
                        << text << " / " << rad::PrettyJson(divisor.value()) << "\n"
                        << FormatValidationErrors(result);
                }
            }
        }
    }
}

TEST(IO, JsonSchemaStandardMetaSchemaDiagnostics)
{
    struct Draft
    {
        rad::JsonSchemaDialect dialect;
        std::string_view uri;
        std::string_view definition;
    };
    constexpr Draft drafts[] = {
        {rad::JsonSchemaDialect::Draft7, "http://json-schema.org/draft-07/schema",
         "/definitions/nonNegativeInteger"},
        {rad::JsonSchemaDialect::Draft2019_09,
         "https://json-schema.org/draft/2019-09/meta/validation", "/$defs/nonNegativeInteger"},
        {rad::JsonSchemaDialect::Draft2020_12,
         "https://json-schema.org/draft/2020-12/meta/validation", "/$defs/nonNegativeInteger"},
    };
    for (const auto& draft : drafts)
    {
        SCOPED_TRACE(draft.uri);
        const auto compiled = rad::JsonSchema::Compile(
            rad::JsonObject{{"$ref", std::string(draft.uri) + "#" + std::string(draft.definition)}},
            draft.dialect);
        ASSERT_TRUE(compiled) << compiled.error().schemaUri << compiled.error().schemaPath << ": "
                              << compiled.error().message;
        const auto invalid = compiled.value().Validate(-1);
        ASSERT_FALSE(invalid);
        ASSERT_EQ(invalid.errors.size(), 1);
        EXPECT_EQ(invalid.errors[0].instancePath, "");
        EXPECT_EQ(invalid.errors[0].schemaPath, std::string(draft.definition) + "/minimum");
        EXPECT_EQ(invalid.errors[0].schemaUri, draft.uri);
    }
}

TEST(IO, JsonSchemaStandardMetaSchemaPrecedence)
{
    struct Draft
    {
        rad::JsonSchemaDialect dialect;
        std::string_view uri;
    };
    constexpr Draft drafts[] = {
        {rad::JsonSchemaDialect::Draft7, "http://json-schema.org/draft-07/schema"},
        {rad::JsonSchemaDialect::Draft2019_09, "https://json-schema.org/draft/2019-09/schema"},
        {rad::JsonSchemaDialect::Draft2020_12, "https://json-schema.org/draft/2020-12/schema"},
    };
    for (const auto& draft : drafts)
    {
        SCOPED_TRACE(draft.uri);
        const auto root = rad::JsonObject{{"$ref", draft.uri}};
        for (const bool alias : {false, true})
        {
            rad::JsonSchemaCompileOptions options;
            options.documents = {
                {alias ? "https://example.com/meta.json" : std::string(draft.uri),
                 rad::JsonObject{{"$id", draft.uri}, {"const", "caller"}}},
            };
            const auto compiled = rad::JsonSchema::Compile(root, draft.dialect, options);
            ASSERT_TRUE(compiled) << compiled.error().message;
            EXPECT_TRUE(compiled.value().Validate("caller"));
            const auto invalid = compiled.value().Validate(rad::JsonObject{});
            ASSERT_FALSE(invalid);
            ASSERT_EQ(invalid.errors.size(), 1);
            EXPECT_EQ(invalid.errors[0].schemaPath, "/const");
            EXPECT_EQ(invalid.errors[0].schemaUri, options.documents[0].uri);
        }
        const auto local = rad::ParseJson(
            std::format(R"json({{"$id": "{}", "type": "object",
                               "properties": {{"value": {{"$ref": "{}"}}}}}})json",
                        draft.uri, draft.uri));
        ASSERT_TRUE(local);
        const auto compiled = rad::JsonSchema::Compile(local.value(), draft.dialect);
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(
            rad::JsonObject{{"value", rad::JsonObject{{"type", 7}}}}));

        if (draft.dialect != rad::JsonSchemaDialect::Draft7)
        {
            rad::JsonSchemaCompileOptions options;
            const auto uri = std::string(draft.uri.substr(0, draft.uri.rfind('/'))) +
                             "/meta/validation";
            options.documents = {
                {uri, rad::JsonObject{{"const", rad::JsonObject{{"type", 42}}}}},
            };
            const auto incomplete = rad::JsonSchema::Compile(root, draft.dialect, options);
            ASSERT_FALSE(incomplete);
            EXPECT_EQ(incomplete.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
            EXPECT_EQ(incomplete.error().schemaUri, draft.uri);
            options.documents[0].schema.as_object()["$defs"] =
                rad::JsonObject{{"stringArray", true}};
            const auto overridden = rad::JsonSchema::Compile(root, draft.dialect, options);
            ASSERT_TRUE(overridden) << overridden.error().message;
            EXPECT_TRUE(overridden.value().Validate(rad::JsonObject{{"type", 42}}));
        }
    }
    const auto mismatched = rad::JsonSchema::Compile(
        rad::JsonObject{{"$ref", "http://json-schema.org/draft-07/schema"}},
        rad::JsonSchemaDialect::Draft2020_12);
    ASSERT_FALSE(mismatched);
    EXPECT_EQ(mismatched.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    EXPECT_EQ(mismatched.error().schemaPath, "/$schema");
    EXPECT_EQ(mismatched.error().schemaUri, "http://json-schema.org/draft-07/schema");
}

TEST(IO, JsonSchemaVocabularyDeclarations)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const std::string prefix = dialect == rad::JsonSchemaDialect::Draft2019_09
                                       ? "https://json-schema.org/draft/2019-09/vocab/"
                                       : "https://json-schema.org/draft/2020-12/vocab/";
        const auto core = prefix + "core";
        rad::JsonSchemaCompileOptions options;
        options.retrievalUri = "https://example.com/root.json";
        const auto base = rad::JsonObject{
            {"$vocabulary", rad::JsonObject{{core, true}, {"urn:example:optional", false}}},
            {"type", "integer"}, {"minimum", 2},
        };
        const auto compiled = rad::JsonSchema::Compile(base, dialect, options);
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(2));
        EXPECT_FALSE(compiled.value().Validate(1));
        EXPECT_FALSE(compiled.value().Validate("invalid"));

        struct Case
        {
            std::string_view uri;
            rad::JsonValue required;
            rad::JsonSchemaCompileErrorCode code;
        };
        const Case cases[] = {
            {"urn:example:required", true, rad::JsonSchemaCompileErrorCode::UnsupportedFeature},
            {"relative", false, rad::JsonSchemaCompileErrorCode::InvalidSchema},
            {"HTTPS://example.com/optional", false, rad::JsonSchemaCompileErrorCode::InvalidSchema},
            {"urn:example:optional", 1, rad::JsonSchemaCompileErrorCode::InvalidSchema},
        };
        for (const auto& testCase : cases)
        {
            SCOPED_TRACE(testCase.uri);
            auto schema = base;
            schema.at("$vocabulary").as_object()[testCase.uri] = testCase.required;
            const auto invalid = rad::JsonSchema::Compile(schema, dialect, options);
            ASSERT_FALSE(invalid);
            EXPECT_EQ(invalid.error().code, testCase.code);
            EXPECT_EQ(invalid.error().schemaPath,
                      testCase.uri.starts_with("HTTPS:")
                          ? "/$vocabulary/HTTPS:~1~1example.com~1optional"
                          : "/$vocabulary/" + std::string(testCase.uri));
            EXPECT_EQ(invalid.error().schemaUri, options.retrievalUri);
        }
        auto format = base;
        const auto formatUri = prefix + (dialect == rad::JsonSchemaDialect::Draft2019_09
                                             ? "format" : "format-assertion");
        format.at("$vocabulary").as_object()[formatUri] = false;
        EXPECT_TRUE(rad::JsonSchema::Compile(format, dialect));
        format.at("$vocabulary").as_object()[formatUri] = true;
        const auto requiredFormat = rad::JsonSchema::Compile(format, dialect);
        ASSERT_FALSE(requiredFormat);
        EXPECT_EQ(requiredFormat.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);

        for (const rad::JsonValue vocabulary :
             {rad::JsonValue(1), rad::JsonValue(rad::JsonObject{}),
              rad::JsonValue(rad::JsonObject{{core, false}})})
        {
            const auto invalid = rad::JsonSchema::Compile(
                rad::JsonObject{{"$vocabulary", vocabulary}}, dialect, options);
            ASSERT_FALSE(invalid);
            EXPECT_EQ(invalid.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
            EXPECT_EQ(invalid.error().schemaPath,
                      vocabulary.is_object() && vocabulary.as_object().contains(core)
                          ? "/$vocabulary/https:~1~1json-schema.org~1draft~1" +
                                std::string(dialect == rad::JsonSchemaDialect::Draft2019_09
                                                ? "2019-09"
                                                : "2020-12") +
                                "~1vocab~1core"
                          : "/$vocabulary");
            EXPECT_EQ(invalid.error().schemaUri, options.retrievalUri);
        }
    }
    const auto ignored = rad::JsonSchema::Compile(
        rad::JsonObject{{"$vocabulary", 42}}, rad::JsonSchemaDialect::Draft7);
    ASSERT_TRUE(ignored) << ignored.error().message;
    EXPECT_TRUE(ignored.value().Validate(1));
}

TEST(IO, JsonSchemaCustomDialectSelection)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const std::string base = dialect == rad::JsonSchemaDialect::Draft2019_09
                                     ? "https://json-schema.org/draft/2019-09/"
                                     : "https://json-schema.org/draft/2020-12/";
        rad::JsonSchemaCompileOptions options;
        options.retrievalUri = "https://example.com/root.json";
        options.documents = {
            {"https://example.com/meta-a.json",
             rad::JsonObject{{"$id", "https://example.com/applicator"},
                             {"$schema", base + "schema"},
                             {"$vocabulary", rad::JsonObject{{base + "vocab/core", true},
                                                            {base + "vocab/applicator", true}}}}},
            {"https://example.com/meta-b.json",
             rad::JsonObject{{"$id", "https://example.com/validation"},
                             {"$schema", base + "schema"},
                             {"$vocabulary", rad::JsonObject{{base + "vocab/core", true},
                                                            {base + "vocab/validation", false}}}}},
        };
        const auto loose = rad::ParseJson(R"json({
            "$schema": "https://example.com/applicator",
            "type": 42, "enum": [], "required": false, "pattern": 1,
            "minimum": false, "minContains": "ignored", "uniqueItems": 1,
            "dependentRequired": 4, "properties": {"value": {"minimum": false}}
        })json");
        ASSERT_TRUE(loose);
        const auto inferred = rad::JsonSchema::Compile(loose.value(), options);
        ASSERT_TRUE(inferred) << inferred.error().schemaUri << inferred.error().schemaPath << ": "
                              << inferred.error().message;
        EXPECT_EQ(inferred.value().Dialect(), dialect);
        EXPECT_TRUE(inferred.value().Validate(rad::JsonObject{{"value", 1}}));
        EXPECT_TRUE(rad::JsonSchema::Compile(loose.value(), dialect, options));

        const auto strict = rad::ParseJson(R"json({
            "$schema": "https://example.com/validation", "type": "integer", "minimum": 1,
            "allOf": 5, "items": 5, "unevaluatedProperties": 5,
            "properties": {"bad": {"$id": 5, "$ref": "unregistered.json"}}
        })json");
        ASSERT_TRUE(strict);
        const auto compiled = rad::JsonSchema::Compile(strict.value(), options);
        ASSERT_TRUE(compiled) << compiled.error().schemaPath << ": " << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(1));
        EXPECT_FALSE(compiled.value().Validate(0));
        EXPECT_FALSE(compiled.value().Validate("invalid"));
        auto malformed = strict.value();
        malformed.as_object()["type"] = 42;
        const auto invalid = rad::JsonSchema::Compile(malformed, options);
        ASSERT_FALSE(invalid);
        EXPECT_EQ(invalid.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
        EXPECT_EQ(invalid.error().schemaPath, "/type");
        EXPECT_EQ(invalid.error().schemaUri, options.retrievalUri);

        auto pointed = strict.value();
        pointed.as_object()["$ref"] = "#/properties/bad";
        const auto badTarget = rad::JsonSchema::Compile(pointed, options);
        ASSERT_FALSE(badTarget);
        EXPECT_EQ(badTarget.error().schemaPath, "/properties/bad/$id");

        options.documents.push_back(
            {"https://example.com/defaults",
             rad::JsonObject{{"$schema", "https://example.com/validation"}}});
        const auto defaults = rad::JsonSchema::Compile(
            rad::JsonObject{{"$schema", "https://example.com/defaults"},
                            {"type", "object"}, {"properties", rad::JsonObject{{"value", false}}}},
            options);
        ASSERT_TRUE(defaults) << defaults.error().message;
        EXPECT_TRUE(defaults.value().Validate(rad::JsonObject{}));
        EXPECT_FALSE(defaults.value().Validate(rad::JsonObject{{"value", 1}}));
        EXPECT_FALSE(defaults.value().Validate(1));
    }
}

TEST(IO, JsonSchemaCustomDialectCandidateShapes)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        const std::string base = dialect == rad::JsonSchemaDialect::Draft2019_09
                                     ? "https://json-schema.org/draft/2019-09/"
                                     : "https://json-schema.org/draft/2020-12/";
        const char* keyword =
            dialect == rad::JsonSchemaDialect::Draft2019_09 ? "items" : "prefixItems";
        const rad::JsonValue meta =
            rad::JsonObject{{"$id", "https://example.com/meta"},
                            {"$schema", base + "schema"},
                            {"$vocabulary", rad::JsonObject{{base + "vocab/core", true},
                                                            {base + "vocab/validation", false}}}};
        rad::JsonSchemaCompileOptions options;
        options.retrievalUri = "https://example.com/root.json";
        options.documents = {
            {"https://example.com/container",
             rad::JsonObject{{"$schema", base + "schema"}, {keyword, rad::JsonArray{meta}}}},
        };
        const rad::JsonValue schema = rad::JsonObject{
            {"$schema", "https://example.com/meta"}, {"type", "integer"}, {"allOf", 5}};
        const auto compiled = rad::JsonSchema::Compile(schema, options);
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(1));
        EXPECT_FALSE(compiled.value().Validate("invalid"));
        options.documents[0]
            .schema.as_object()
            .at(keyword)
            .as_array()[0]
            .as_object()["$vocabulary"] = rad::JsonObject{};
        const auto missingCore = rad::JsonSchema::Compile(schema, options);
        ASSERT_FALSE(missingCore);
        EXPECT_EQ(missingCore.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
        EXPECT_EQ(missingCore.error().schemaPath,
                  std::string("/") + keyword + "/0/$vocabulary/https:~1~1json-schema.org~1draft~1" +
                      (dialect == rad::JsonSchemaDialect::Draft2019_09 ? "2019-09" : "2020-12") +
                      "~1vocab~1core");
        EXPECT_EQ(missingCore.error().schemaUri, options.documents[0].uri);
        if (dialect == rad::JsonSchemaDialect::Draft2020_12)
        {
            options.documents[0].schema.as_object().erase("prefixItems");
            options.documents[0].schema.as_object()["items"] = rad::JsonArray{meta};
            const auto hidden = rad::JsonSchema::Compile(schema, options);
            ASSERT_FALSE(hidden);
            EXPECT_EQ(hidden.error().schemaPath, "/$schema");
            EXPECT_EQ(hidden.error().schemaUri, options.retrievalUri);
        }
    }
}

TEST(IO, JsonSchemaCustomDialectResourceProfiles)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        const std::string base = dialect == rad::JsonSchemaDialect::Draft2019_09
                                     ? "https://json-schema.org/draft/2019-09/"
                                     : "https://json-schema.org/draft/2020-12/";
        rad::JsonSchemaCompileOptions options;
        options.retrievalUri = "https://example.com/root.json";
        options.documents = {
            {"https://example.com/loose-meta",
             rad::JsonObject{{"$schema", base + "schema"},
                             {"$vocabulary", rad::JsonObject{{base + "vocab/core", true},
                                                            {base + "vocab/applicator", true}}}}},
        };
        const auto schema = rad::ParseJson(std::format(R"json({{
            "$schema": "https://example.com/loose-meta",
            "$defs": {{
                "strict": {{"$id": "strict", "$schema": "{}schema", "type": "integer"}},
                "loose": {{"$id": "loose", "type": 42}}
            }},
            "properties": {{
                "strict": {{"$ref": "strict"}},
                "loose": {{"$ref": "loose"}}
            }}
        }})json", base));
        ASSERT_TRUE(schema);
        const auto compiled = rad::JsonSchema::Compile(schema.value(), options);
        ASSERT_TRUE(compiled) << compiled.error().schemaPath << ": " << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(rad::JsonObject{{"strict", 1}, {"loose", "ignored"}}));
        const auto invalid = compiled.value().Validate(rad::JsonObject{{"strict", "invalid"}});
        ASSERT_FALSE(invalid);
        ASSERT_EQ(invalid.errors.size(), 1);
        EXPECT_EQ(invalid.errors[0].instancePath, "/strict");
        EXPECT_EQ(invalid.errors[0].schemaPath, "/$defs/strict/type");
        EXPECT_EQ(invalid.errors[0].schemaUri, options.retrievalUri);

        auto nonResource = schema.value();
        nonResource.as_object().at("properties").as_object().at("strict") =
            rad::JsonObject{{"$schema", base + "schema"}, {"type", "integer"}};
        const auto badDeclaration = rad::JsonSchema::Compile(nonResource, options);
        ASSERT_FALSE(badDeclaration);
        EXPECT_EQ(badDeclaration.error().schemaPath, "/properties/strict/$schema");
        EXPECT_EQ(badDeclaration.error().schemaUri, options.retrievalUri);

        options.documents.push_back(
            {"https://example.com/foreign.json",
             rad::JsonObject{{"$schema", base + "schema"}, {"type", "integer"}}});
        auto referenced = schema.value();
        referenced.as_object()["$ref"] = "foreign.json";
        const auto external = rad::JsonSchema::Compile(referenced, options);
        ASSERT_TRUE(external) << external.error().message;
        EXPECT_TRUE(external.value().Validate(1));
        EXPECT_FALSE(external.value().Validate("invalid"));
        options.documents.back().schema.as_object()["$schema"] = "https://example.com/missing";
        const auto badExternal = rad::JsonSchema::Compile(referenced, options);
        ASSERT_FALSE(badExternal);
        EXPECT_EQ(badExternal.error().schemaPath, "/$schema");
        EXPECT_EQ(badExternal.error().schemaUri, options.documents.back().uri);
    }
}

TEST(IO, JsonSchemaCustomDialectUnevaluatedVocabulary)
{
    rad::JsonSchemaCompileOptions options;
    options.documents = {
        {"https://example.com/meta",
         rad::JsonObject{{"$schema", "https://json-schema.org/draft/2020-12/schema"},
                         {"$vocabulary", rad::JsonObject{
                             {"https://json-schema.org/draft/2020-12/vocab/core", true},
                             {"https://json-schema.org/draft/2020-12/vocab/unevaluated", false}}}}},
    };
    const auto compiled = rad::JsonSchema::Compile(
        rad::JsonObject{{"$schema", "https://example.com/meta"},
                        {"properties", rad::JsonObject{{"value", true}}},
                        {"prefixItems", rad::JsonArray{true}},
                        {"unevaluatedProperties", false}, {"unevaluatedItems", false}},
        options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    EXPECT_TRUE(compiled.value().Validate(rad::JsonObject{}));
    EXPECT_TRUE(compiled.value().Validate(rad::JsonArray{}));
    EXPECT_FALSE(compiled.value().Validate(rad::JsonObject{{"value", 1}}));
    EXPECT_FALSE(compiled.value().Validate(rad::JsonArray{1}));
}

TEST(IO, JsonSchemaCustomDialectDiagnostics)
{
    constexpr auto dialect = rad::JsonSchemaDialect::Draft2020_12;
    rad::JsonSchemaCompileOptions options;
    options.retrievalUri = "https://example.com/root.json";
    const auto root = rad::JsonObject{{"$schema", "https://example.com/meta"}};
    const auto missing = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().schemaPath, "/$schema");
    EXPECT_EQ(missing.error().schemaUri, options.retrievalUri);
    options.documents = {
        {"https://example.com/meta",
         rad::JsonObject{{"$schema", "https://example.com/other"}}},
        {"https://example.com/other",
         rad::JsonObject{{"$schema", "https://example.com/meta"}}},
    };
    const auto cyclic = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(cyclic);
    EXPECT_EQ(cyclic.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    EXPECT_EQ(cyclic.error().schemaPath, "/$schema");
    options.documents = {
        {"https://example.com/meta", rad::JsonObject{
            {"$schema", "https://json-schema.org/draft/2020-12/schema"},
            {"$vocabulary", rad::JsonObject{
                {"https://json-schema.org/draft/2020-12/vocab/core", true},
                {"urn:example:required", true}}},
        }},
    };
    const auto unsupported = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(unsupported);
    EXPECT_EQ(unsupported.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    EXPECT_EQ(unsupported.error().schemaPath, "/$vocabulary/urn:example:required");
    EXPECT_EQ(unsupported.error().schemaUri, options.documents[0].uri);
    options.documents[0].schema.as_object().at("$vocabulary").as_object()
        ["urn:example:required"] = false;
    EXPECT_TRUE(rad::JsonSchema::Compile(root, options));
    const auto mismatched = rad::JsonSchema::Compile(
        root, rad::JsonSchemaDialect::Draft2019_09, options);
    ASSERT_FALSE(mismatched);
    EXPECT_EQ(mismatched.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    EXPECT_TRUE(rad::JsonSchema::Compile(root, dialect, options));
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

    rad::JsonSchemaCompileOptions compileOptions;
    const auto remotes = suiteRoot / "remotes";
    ASSERT_TRUE(std::filesystem::is_directory(remotes)) << remotes.string();
    for (const auto& entry : std::filesystem::recursive_directory_iterator(remotes))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".json")
        {
            continue;
        }
        const auto text = rad::File::ReadAllText(entry.path());
        ASSERT_TRUE(text) << entry.path().string();
        const auto schema = rad::ParseJson(*text);
        ASSERT_TRUE(schema) << entry.path().string();
        compileOptions.documents.push_back({
            "http://localhost:1234/" +
                std::filesystem::relative(entry.path(), remotes).generic_string(),
            schema.value(),
        });
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
        for (const auto* name : {"ecmascript-regex.json", "non-bmp-regex.json",
                                 "float-overflow.json"})
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
                    rad::JsonSchema::Compile(group.at("schema"), suite.dialect, compileOptions);
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
                    ADD_FAILURE() << compiled.error().schemaUri
                                  << compiled.error().schemaPath << ": "
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
