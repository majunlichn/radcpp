#include <rad/IO/File.h>
#include <rad/IO/Json.h>

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

        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(testsDirectory))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".json")
            {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());

        for (const auto& file : files)
        {
            SCOPED_TRACE(file.string());
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
                if (!compiled)
                {
                    const bool knownInvalidSchema =
                        compiled.error().code ==
                            rad::JsonSchemaCompileErrorCode::InvalidSchema &&
                        IsKnownInvalidOfficialSchema(group.at("schema"));
                    if (compiled.error().code !=
                            rad::JsonSchemaCompileErrorCode::UnsupportedFeature &&
                        !knownInvalidSchema)
                    {
                        ADD_FAILURE() << compiled.error().schemaPath << ": "
                                      << compiled.error().message;
                    }
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
                    EXPECT_EQ(static_cast<bool>(result), expected)
                        << std::format("data: {}\n{}",
                                       rad::PrettyJson(test.at("data")),
                                       FormatValidationErrors(result));
                }
            }
        }
        EXPECT_GT(suiteExecutedCases, 0) << testsDirectory.string();
    }

    if (availableSuites == 0)
    {
        GTEST_SKIP() << "No supported schema test directories were found";
    }
    EXPECT_GT(executedCases, 0);
}
