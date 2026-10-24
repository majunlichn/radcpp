#include <rad/IO/File.h>
#include <rad/IO/Json.h>

#include "JsonSchema.test.h"

#include <boost/filesystem/operations.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <string_view>
#include <cstddef>

using rad::test::FormatValidationErrors;

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

TEST(IO, JsonSchemaCompileFileOptions)
{
    const auto path = std::filesystem::temp_directory_path() /
                      boost::filesystem::unique_path("rad-schema-%%%%-%%%%-%%%%.json").string();
    ASSERT_TRUE(rad::File::WriteText(path, R"json({"$ref": "child.json"})json"));
    struct SchemaFile
    {
        rad::FilePath path;
        ~SchemaFile() { EXPECT_TRUE(rad::File::Remove(path)); }
    } file{path};
    const rad::JsonSchemaCompileOptions options{
        .dialect = rad::JsonSchemaDialect::Draft2020_12,
        .retrievalUri = "https://example.com/root.json",
        .documents =
            {
                {"https://example.com/child.json", rad::JsonObject{{"type", "integer"}}},
            },
    };
    const auto compiled = rad::JsonSchema::CompileFile(path, options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    EXPECT_EQ(compiled.value().Dialect(), options.dialect);
    EXPECT_TRUE(compiled.value().Validate(1));
    EXPECT_FALSE(compiled.value().Validate("invalid"));

    ASSERT_TRUE(rad::File::WriteText(path, "{"));
    const auto invalidJson = rad::JsonSchema::CompileFile(path, options);
    ASSERT_FALSE(invalidJson);
    EXPECT_EQ(invalidJson.error().code, rad::JsonSchemaCompileErrorCode::InvalidJson);
    EXPECT_EQ(invalidJson.error().dialect, options.dialect);
    EXPECT_EQ(invalidJson.error().schemaUri, options.retrievalUri);

    const auto missing = rad::JsonSchema::CompileFile(path.string() + ".missing", options);
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().code, rad::JsonSchemaCompileErrorCode::FileReadError);
    EXPECT_EQ(missing.error().dialect, options.dialect);
    EXPECT_EQ(missing.error().schemaUri, options.retrievalUri);
}

TEST(IO, JsonSchemaDiagnostic)
{
    const rad::JsonSchemaDiagnostic diagnostic{
        "/value", "https://example.com/schema.json", "/properties/value/type", "expected integer"};
    EXPECT_EQ(diagnostic.instancePath, "/value");
    EXPECT_EQ(diagnostic.schemaUri, "https://example.com/schema.json");
    EXPECT_EQ(diagnostic.schemaPath, "/properties/value/type");
    EXPECT_EQ(diagnostic.message, "expected integer");

    rad::JsonSchemaValidationResult result;
    result.errors.push_back(diagnostic);
    result.status = rad::JsonSchemaValidationStatus::InternalError;
    result.evaluationError = diagnostic;
    EXPECT_EQ(result.errors[0].schemaUri, diagnostic.schemaUri);
    EXPECT_EQ(result.evaluationError->schemaPath, diagnostic.schemaPath);
}

TEST(IO, JsonSchemaValidationStatus)
{
    rad::JsonSchemaValidationResult empty;
    EXPECT_EQ(empty.status, rad::JsonSchemaValidationStatus::Complete);
    EXPECT_TRUE(empty);
    for (const auto status : {rad::JsonSchemaValidationStatus::DepthLimitExceeded,
                              rad::JsonSchemaValidationStatus::RegexResourceLimitExceeded,
                              rad::JsonSchemaValidationStatus::RegexEvaluationFailed,
                              rad::JsonSchemaValidationStatus::InternalError})
    {
        empty.status = status;
        EXPECT_FALSE(empty);
    }

    const auto compiled = rad::JsonSchema::Compile(
        rad::JsonObject{{"type", "integer"}}, {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_TRUE(compiled);
    const auto valid = compiled.value().Validate(1);
    EXPECT_TRUE(valid);
    EXPECT_EQ(valid.status, rad::JsonSchemaValidationStatus::Complete);
    EXPECT_TRUE(valid.errors.empty());
    EXPECT_FALSE(valid.evaluationError);

    for (const std::size_t maxErrors : {0U, 1U, 2U})
    {
        rad::JsonSchemaValidationOptions options;
        options.maxErrors = maxErrors;
        const auto invalid = compiled.value().Validate("invalid", options);
        EXPECT_FALSE(invalid);
        EXPECT_EQ(invalid.status, rad::JsonSchemaValidationStatus::Complete);
        EXPECT_EQ(invalid.errors.size(), 1);
        EXPECT_FALSE(invalid.evaluationError);
    }

    const auto repeated =
        rad::JsonSchema::Compile(rad::JsonObject{{"allOf", rad::JsonArray{false, false, false}}},
                                 {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_TRUE(repeated);
    rad::JsonSchemaValidationOptions options;
    options.maxErrors = 1;
    const auto truncated = repeated.value().Validate(1, options);
    EXPECT_FALSE(truncated);
    EXPECT_EQ(truncated.status, rad::JsonSchemaValidationStatus::Complete);
    EXPECT_EQ(truncated.errors.size(), 1);
    EXPECT_FALSE(truncated.evaluationError);
}

TEST(IO, JsonSchemaEvaluationFailurePropagation)
{
    constexpr std::string_view schemas[] = {
        R"json({"allOf": [{"allOf": [true]}]})json",
        R"json({"not": {"allOf": [true]}})json",
        R"json({"anyOf": [{"allOf": [true]}, true]})json",
        R"json({"oneOf": [{"allOf": [true]}, true]})json",
        R"json({"if": {"allOf": [true]}, "then": true, "else": true})json",
        R"json({"contains": {"allOf": [true]}})json",
        R"json({"anyOf": [{"not": {"allOf": [true]}}, true]})json",
    };
    for (const auto source : schemas)
    {
        SCOPED_TRACE(source);
        const auto schema = rad::ParseJson(source);
        ASSERT_TRUE(schema);
        const auto compiled = rad::JsonSchema::Compile(
            schema.value(), {.dialect = rad::JsonSchemaDialect::Draft2020_12});
        ASSERT_TRUE(compiled);
        rad::JsonSchemaValidationOptions options;
        options.maxDepth = 1;
        const auto result = compiled.value().Validate(rad::JsonArray{1}, options);
        EXPECT_FALSE(result);
        EXPECT_EQ(result.status, rad::JsonSchemaValidationStatus::DepthLimitExceeded);
        EXPECT_TRUE(result.errors.empty());
        ASSERT_TRUE(result.evaluationError);
        EXPECT_EQ(result.evaluationError->message, "maximum validation depth exceeded");
        EXPECT_EQ(compiled.value().Validate(rad::JsonArray{1}).status,
                  rad::JsonSchemaValidationStatus::Complete);
    }

    const auto schema =
        rad::ParseJson(R"json({"allOf": [false, {"allOf": [true]}, {"allOf": [false]}]})json");
    ASSERT_TRUE(schema);
    const auto compiled =
        rad::JsonSchema::Compile(schema.value(), {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_TRUE(compiled);
    for (const std::size_t maxErrors : {0U, 1U, 2U})
    {
        rad::JsonSchemaValidationOptions options;
        options.maxDepth = 1;
        options.maxErrors = maxErrors;
        const auto result = compiled.value().Validate(1, options);
        EXPECT_FALSE(result);
        EXPECT_EQ(result.status, rad::JsonSchemaValidationStatus::DepthLimitExceeded);
        ASSERT_EQ(result.errors.size(), 1);
        EXPECT_EQ(result.errors[0].schemaPath, "/allOf/0");
        ASSERT_TRUE(result.evaluationError);
        EXPECT_EQ(result.evaluationError->schemaPath, "/allOf/1/allOf/0");
    }
}

TEST(IO, JsonSchemaResourceFailureWithErrorLimit)
{
    const auto schema = rad::ParseJson(
        R"json({"anyOf": [
            {"type": "integer", "properties": {"child": {"allOf": [true]}}},
            true
        ]})json");
    ASSERT_TRUE(schema);
    const auto compiled =
        rad::JsonSchema::Compile(schema.value(), {.dialect = rad::JsonSchemaDialect::Draft2020_12});
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
        EXPECT_EQ(result.status, rad::JsonSchemaValidationStatus::DepthLimitExceeded);
        EXPECT_TRUE(result.errors.empty());
        ASSERT_TRUE(result.evaluationError);
        EXPECT_EQ(result.evaluationError->instancePath, "/child");
        EXPECT_EQ(result.evaluationError->schemaPath, "/anyOf/0/properties/child/allOf/0");
        EXPECT_EQ(result.evaluationError->message, "maximum validation depth exceeded");
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
        const auto compiled = rad::JsonSchema::Compile(schema.value(), {.dialect = dialect});
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
        EXPECT_EQ(limited.status, rad::JsonSchemaValidationStatus::DepthLimitExceeded);
        EXPECT_TRUE(limited.errors.empty());
        ASSERT_TRUE(limited.evaluationError);
        EXPECT_EQ(limited.evaluationError->instancePath, "/0");
        EXPECT_EQ(limited.evaluationError->schemaPath, "/unevaluatedItems");
        EXPECT_EQ(limited.evaluationError->message, "maximum validation depth exceeded");

        const auto invalid = rad::JsonSchema::Compile(rad::JsonObject{{"unevaluatedItems", 42}},
                                                      {.dialect = dialect});
        ASSERT_FALSE(invalid);
        EXPECT_EQ(invalid.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
        EXPECT_EQ(invalid.error().schemaPath, "/unevaluatedItems");
    }
    const auto olderDraft = rad::JsonSchema::Compile(rad::JsonObject{{"unevaluatedItems", 42}},
                                                     {.dialect = rad::JsonSchemaDialect::Draft7});
    ASSERT_TRUE(olderDraft) << olderDraft.error().message;
    EXPECT_TRUE(olderDraft.value().Validate(rad::JsonArray{1, "ignored"}));
}
