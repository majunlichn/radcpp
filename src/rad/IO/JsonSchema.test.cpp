#include <rad/IO/File.h>
#include <rad/IO/JsonSchema.h>

#include "JsonSchema.test.h"

#include <boost/filesystem/operations.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <string_view>
#include <cstddef>
#include <utility>

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
    const auto compiled = rad::JsonSchema::CompileFile(path, {}, options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    EXPECT_EQ(compiled.value().Dialect(), options.dialect);
    EXPECT_TRUE(compiled.value().Validate(1));
    EXPECT_FALSE(compiled.value().Validate("invalid"));

    ASSERT_TRUE(rad::File::WriteText(path, R"json({
        // Schema files can opt into non-standard JSON syntax.
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "integer",
    })json"));
    const auto strict = rad::JsonSchema::CompileFile(path);
    ASSERT_FALSE(strict);
    EXPECT_EQ(strict.error().code, rad::JsonSchemaCompileErrorCode::InvalidJson);
    rad::JsonParseOptions parseOptions;
    parseOptions.allow_comments = true;
    parseOptions.allow_trailing_commas = true;
    const auto parsed = rad::JsonSchema::CompileFile(path, parseOptions);
    ASSERT_TRUE(parsed) << parsed.error().message;
    EXPECT_TRUE(parsed.value().Validate(1));
    EXPECT_FALSE(parsed.value().Validate("invalid"));

    rad::JsonValue nested = true;
    for (std::size_t index = 0; index < 40; ++index)
    {
        nested = rad::JsonObject{{"allOf", rad::JsonArray{std::move(nested)}}};
    }
    ASSERT_TRUE(rad::File::WriteText(path, rad::PrettyJson(nested)));
    const auto parseLimited = rad::JsonSchema::CompileFile(path, {}, options);
    ASSERT_FALSE(parseLimited);
    EXPECT_EQ(parseLimited.error().code, rad::JsonSchemaCompileErrorCode::InvalidJson);
    parseOptions.max_depth = 128;
    auto compileOptions = options;
    compileOptions.maxCompilationDepth = 39;
    const auto compileLimited =
        rad::JsonSchema::CompileFile(path, parseOptions, compileOptions);
    ASSERT_FALSE(compileLimited);
    EXPECT_EQ(compileLimited.error().code,
              rad::JsonSchemaCompileErrorCode::DepthLimitExceeded);
    EXPECT_EQ(compileLimited.error().schemaUri, options.retrievalUri);
    compileOptions.maxCompilationDepth = 40;
    const auto deep = rad::JsonSchema::CompileFile(path, parseOptions, compileOptions);
    ASSERT_TRUE(deep) << deep.error().message;
    EXPECT_TRUE(deep.value().Validate(1));

    ASSERT_TRUE(rad::File::WriteText(path, "{"));
    const auto invalidJson = rad::JsonSchema::CompileFile(path, {}, options);
    ASSERT_FALSE(invalidJson);
    EXPECT_EQ(invalidJson.error().code, rad::JsonSchemaCompileErrorCode::InvalidJson);
    EXPECT_EQ(invalidJson.error().schemaUri, options.retrievalUri);

    const auto missing = rad::JsonSchema::CompileFile(path.string() + ".missing", {}, options);
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().code, rad::JsonSchemaCompileErrorCode::FileReadFailed);
    EXPECT_EQ(missing.error().schemaUri, options.retrievalUri);
}

TEST(IO, JsonSchemaCompilationDepth)
{
    EXPECT_EQ(rad::JsonSchemaCompileOptions{}.maxCompilationDepth, 128);
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        rad::JsonSchemaCompileOptions options{.dialect = dialect};
        options.retrievalUri = "https://example.com/root.json";
        options.maxCompilationDepth = 0;
        EXPECT_TRUE(rad::JsonSchema::Compile(true, options));
        EXPECT_TRUE(rad::JsonSchema::Compile(rad::JsonObject{{"type", "integer"}}, options));
        const rad::JsonValue nested = rad::JsonObject{{"allOf", rad::JsonArray{true, true, true}}};
        const auto limited = rad::JsonSchema::Compile(nested, options);
        ASSERT_FALSE(limited);
        EXPECT_EQ(limited.error().code, rad::JsonSchemaCompileErrorCode::DepthLimitExceeded);
        EXPECT_EQ(limited.error().schemaUri, options.retrievalUri);
        EXPECT_EQ(limited.error().schemaPath, "/allOf/0");
        options.maxCompilationDepth = 1;
        EXPECT_TRUE(rad::JsonSchema::Compile(nested, options));

        options.documents = {
            {"https://example.com/a.json", rad::JsonObject{{"$ref", "b.json"}}},
            {"https://example.com/b.json", rad::JsonObject{{"$ref", "c.json"}}},
            {"https://example.com/c.json", rad::JsonObject{{"type", "integer"}}},
        };
        const rad::JsonValue referenced = rad::JsonObject{{"$ref", "a.json"}};
        options.maxCompilationDepth = 2;
        const auto referenceLimited = rad::JsonSchema::Compile(referenced, options);
        ASSERT_FALSE(referenceLimited);
        EXPECT_EQ(referenceLimited.error().code,
                  rad::JsonSchemaCompileErrorCode::DepthLimitExceeded);
        EXPECT_EQ(referenceLimited.error().schemaUri, "https://example.com/c.json");
        EXPECT_EQ(referenceLimited.error().schemaPath, "");
        options.maxCompilationDepth = 3;
        const auto compiled = rad::JsonSchema::Compile(referenced, options);
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(1));
        rad::JsonSchemaValidationOptions validationOptions;
        validationOptions.maxEvaluationDepth = 2;
        const auto evaluationLimited = compiled.value().Validate(1, validationOptions);
        ASSERT_FALSE(evaluationLimited);
        ASSERT_EQ(evaluationLimited.errors.size(), 1);
        EXPECT_EQ(evaluationLimited.errors[0].errorCode,
                  rad::JsonSchemaValidationErrorCode::DepthLimitExceeded);
        validationOptions.maxEvaluationDepth = 3;
        EXPECT_TRUE(compiled.value().Validate(1, validationOptions));
    }
}

TEST(IO, JsonSchemaDefaultCompilationDepth)
{
    for (const std::string_view keyword : {"allOf", "properties", "items", "dependentSchemas"})
    {
        SCOPED_TRACE(keyword);
        const auto wrap = [keyword](rad::JsonValue child) -> rad::JsonValue
        {
            if (keyword == "allOf")
            {
                return rad::JsonObject{{keyword, rad::JsonArray{std::move(child)}}};
            }
            if (keyword == "properties" || keyword == "dependentSchemas")
            {
                return rad::JsonObject{{keyword, rad::JsonObject{{"child", std::move(child)}}}};
            }
            return rad::JsonObject{{keyword, std::move(child)}};
        };
        rad::JsonValue nested = true;
        for (std::size_t index = 0; index < 128; ++index)
        {
            nested = wrap(std::move(nested));
        }
        rad::JsonSchemaCompileOptions options{.dialect = rad::JsonSchemaDialect::Draft2020_12};
        EXPECT_TRUE(rad::JsonSchema::Compile(nested, options));
        nested = wrap(std::move(nested));
        const auto limited = rad::JsonSchema::Compile(nested, options);
        ASSERT_FALSE(limited);
        EXPECT_EQ(limited.error().code, rad::JsonSchemaCompileErrorCode::DepthLimitExceeded);
        options.maxCompilationDepth = 129;
        EXPECT_TRUE(rad::JsonSchema::Compile(nested, options));
    }
}

TEST(IO, JsonSchemaErrors)
{
    const rad::JsonSchemaCompileError compileError{
        rad::JsonSchemaCompileErrorCode::InvalidSchema, "https://example.com/schema.json",
        "/properties/value/type", "type must be a known type name"};
    EXPECT_EQ(compileError.code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    EXPECT_EQ(compileError.schemaUri, "https://example.com/schema.json");
    EXPECT_EQ(compileError.schemaPath, "/properties/value/type");
    EXPECT_EQ(compileError.message, "type must be a known type name");

    const rad::JsonSchemaValidationError validationError{
        rad::JsonSchemaValidationErrorCode::SchemaViolation, "/value",
        "https://example.com/schema.json", "/properties/value/type", "expected integer"};
    EXPECT_EQ(validationError.errorCode, rad::JsonSchemaValidationErrorCode::SchemaViolation);
    EXPECT_EQ(validationError.instancePath, "/value");
    EXPECT_EQ(validationError.schemaUri, "https://example.com/schema.json");
    EXPECT_EQ(validationError.schemaPath, "/properties/value/type");
    EXPECT_EQ(validationError.message, "expected integer");

    rad::JsonSchemaValidationResult result;
    result.errors.push_back(validationError);
    EXPECT_EQ(result.errors[0].errorCode, rad::JsonSchemaValidationErrorCode::SchemaViolation);
    EXPECT_EQ(result.errors[0].schemaUri, validationError.schemaUri);
    result.errors.push_back({rad::JsonSchemaValidationErrorCode::InternalError,
                             validationError.instancePath, validationError.schemaUri,
                             validationError.schemaPath, validationError.message});
    EXPECT_EQ(result.errors[1].errorCode, rad::JsonSchemaValidationErrorCode::InternalError);
    EXPECT_EQ(result.errors[1].schemaPath, validationError.schemaPath);
}

TEST(IO, JsonSchemaValidationResult)
{
    rad::JsonSchemaValidationResult empty;
    EXPECT_TRUE(empty);
    for (const auto errorCode : {rad::JsonSchemaValidationErrorCode::SchemaViolation,
                                 rad::JsonSchemaValidationErrorCode::DepthLimitExceeded,
                                 rad::JsonSchemaValidationErrorCode::RegexResourceLimitExceeded,
                                 rad::JsonSchemaValidationErrorCode::RegexEvaluationFailed,
                                 rad::JsonSchemaValidationErrorCode::InternalError})
    {
        rad::JsonSchemaValidationResult result;
        result.errors.push_back({errorCode, {}, {}, {}, {}});
        EXPECT_FALSE(result);
    }

    const auto compiled = rad::JsonSchema::Compile(
        rad::JsonObject{{"type", "integer"}}, {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_TRUE(compiled);
    const auto valid = compiled.value().Validate(1);
    EXPECT_TRUE(valid);
    EXPECT_TRUE(valid.errors.empty());

    for (const std::size_t maxErrors : {0U, 1U, 2U})
    {
        rad::JsonSchemaValidationOptions options;
        options.maxErrors = maxErrors;
        const auto invalid = compiled.value().Validate("invalid", options);
        EXPECT_FALSE(invalid);
        ASSERT_EQ(invalid.errors.size(), 1);
        EXPECT_EQ(invalid.errors[0].errorCode, rad::JsonSchemaValidationErrorCode::SchemaViolation);
    }

    const auto repeated =
        rad::JsonSchema::Compile(rad::JsonObject{{"allOf", rad::JsonArray{false, false, false}}},
                                 {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_TRUE(repeated);
    rad::JsonSchemaValidationOptions options;
    options.maxErrors = 1;
    const auto truncated = repeated.value().Validate(1, options);
    EXPECT_FALSE(truncated);
    ASSERT_EQ(truncated.errors.size(), 1);
    EXPECT_EQ(truncated.errors[0].errorCode, rad::JsonSchemaValidationErrorCode::SchemaViolation);
}

TEST(IO, JsonSchemaEvaluationDepth)
{
    EXPECT_EQ(rad::JsonSchemaValidationOptions{}.maxEvaluationDepth, 128);
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const auto compiled = rad::JsonSchema::Compile(
            rad::JsonObject{{"items", rad::JsonObject{{"$ref", "#"}}}}, {.dialect = dialect});
        ASSERT_TRUE(compiled) << compiled.error().message;

        rad::JsonSchemaValidationOptions options;
        options.maxEvaluationDepth = 0;
        EXPECT_TRUE(compiled.value().Validate(1, options));
        const auto rootOnly = compiled.value().Validate(rad::JsonArray{1}, options);
        ASSERT_FALSE(rootOnly);
        ASSERT_EQ(rootOnly.errors.size(), 1);
        EXPECT_EQ(rootOnly.errors[0].errorCode,
                  rad::JsonSchemaValidationErrorCode::DepthLimitExceeded);
        EXPECT_EQ(rootOnly.errors[0].instancePath, "/0");
        EXPECT_EQ(rootOnly.errors[0].schemaPath, "/items");

        options.maxEvaluationDepth = 1;
        const auto referenceLimited = compiled.value().Validate(rad::JsonArray{1}, options);
        ASSERT_FALSE(referenceLimited);
        ASSERT_EQ(referenceLimited.errors.size(), 1);
        EXPECT_EQ(referenceLimited.errors[0].errorCode,
                  rad::JsonSchemaValidationErrorCode::DepthLimitExceeded);
        EXPECT_EQ(referenceLimited.errors[0].schemaPath, "");

        options.maxEvaluationDepth = 2;
        EXPECT_TRUE(compiled.value().Validate(rad::JsonArray{1, 2, 3}, options));

        rad::JsonValue instance = 1;
        for (std::size_t index = 0; index < 64; ++index)
        {
            instance = rad::JsonArray{std::move(instance)};
        }
        EXPECT_TRUE(compiled.value().Validate(instance));
        instance = rad::JsonArray{std::move(instance)};
        const auto defaultLimited = compiled.value().Validate(instance);
        ASSERT_FALSE(defaultLimited);
        ASSERT_EQ(defaultLimited.errors.size(), 1);
        EXPECT_EQ(defaultLimited.errors[0].errorCode,
                  rad::JsonSchemaValidationErrorCode::DepthLimitExceeded);

        options.maxEvaluationDepth = 130;
        EXPECT_TRUE(compiled.value().Validate(instance, options));
    }
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
        for (const std::size_t maxErrors : {0U, 1U, 2U})
        {
            rad::JsonSchemaValidationOptions options;
            options.maxEvaluationDepth = 1;
            options.maxErrors = maxErrors;
            const auto result = compiled.value().Validate(rad::JsonArray{1}, options);
            EXPECT_FALSE(result);
            ASSERT_EQ(result.errors.size(), 1);
            EXPECT_EQ(result.errors[0].errorCode,
                      rad::JsonSchemaValidationErrorCode::DepthLimitExceeded);
            EXPECT_EQ(result.errors[0].message, "maximum validation depth exceeded");
        }
        for (const auto& error : compiled.value().Validate(rad::JsonArray{1}).errors)
        {
            EXPECT_EQ(error.errorCode, rad::JsonSchemaValidationErrorCode::SchemaViolation);
        }
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
        options.maxEvaluationDepth = 1;
        options.maxErrors = maxErrors;
        const auto result = compiled.value().Validate(1, options);
        EXPECT_FALSE(result);
        ASSERT_EQ(result.errors.size(), maxErrors <= 1 ? 1 : 2);
        EXPECT_EQ(result.errors[0].errorCode, rad::JsonSchemaValidationErrorCode::SchemaViolation);
        EXPECT_EQ(result.errors[0].schemaPath, "/allOf/0");
        if (maxErrors > 1)
        {
            EXPECT_EQ(result.errors.back().errorCode,
                      rad::JsonSchemaValidationErrorCode::DepthLimitExceeded);
            EXPECT_EQ(result.errors.back().schemaPath, "/allOf/1/allOf/0");
        }
    }
}

TEST(IO, JsonSchemaBranchErrorLimit)
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
    options.maxEvaluationDepth = 2;
    for (const std::size_t maxErrors : {0U, 1U, 2U})
    {
        SCOPED_TRACE(maxErrors);
        options.maxErrors = maxErrors;
        const auto result = compiled.value().Validate(
            rad::JsonObject{{"child", rad::JsonObject{}}}, options);
        EXPECT_TRUE(result);
        EXPECT_TRUE(result.errors.empty());
    }
    options.maxEvaluationDepth = 3;
    EXPECT_TRUE(compiled.value().Validate(
        rad::JsonObject{{"child", rad::JsonObject{}}}, options));
}

TEST(IO, JsonSchemaErrorLimitStopsEvaluation)
{
    struct Case
    {
        std::string_view schema;
        std::string_view instance;
        rad::JsonSchemaDialect dialect = rad::JsonSchemaDialect::Draft2020_12;
    };
    constexpr Case cases[] = {
        {R"json({"allOf": [false, false, {"allOf": [true]}]})json", "1"},
        {R"json({"enum": [2], "const": 3, "allOf": [{"allOf": [true]}]})json", "1"},
        {R"json({"required": ["a", "b"],
                 "properties": {"child": {"allOf": [true]}}})json",
         R"json({"child": {}})json"},
        {R"json({"properties": {"a": false, "b": false, "c": {"allOf": [true]}}})json",
         R"json({"a": 1, "b": 2, "c": {}})json"},
        {R"json({"patternProperties":
                 {"^a$": false, "^b$": false, "^c$": {"allOf": [true]}}})json",
         R"json({"a": 1, "b": 2, "c": {}})json"},
        {R"json({"propertyNames": false, "additionalProperties": {"allOf": [true]}})json",
         R"json({"a": 1, "b": 2})json"},
        {R"json({"additionalProperties": false,
                 "dependentSchemas": {"a": {"allOf": [true]}}})json",
         R"json({"a": 1, "b": 2})json"},
        {R"json({"dependentRequired": {"a": ["x", "y"]},
                 "dependentSchemas": {"a": {"allOf": [true]}}})json",
         R"json({"a": 1})json"},
        {R"json({"dependencies": {"a": ["x", "y"], "b": {"allOf": [true]}}})json",
         R"json({"a": 1, "b": 2})json", rad::JsonSchemaDialect::Draft7},
        {R"json({"unevaluatedProperties":
                 {"type": "object", "properties": {"child": true}}})json",
         R"json({"a": "x", "b": "x", "c": {"child": {}}})json"},
        {R"json({"prefixItems": [false, false, {"allOf": [true]}]})json", "[1, 2, 3]"},
        {R"json({"items": [false, false], "additionalItems": {"allOf": [true]}})json", "[1, 2, 3]",
         rad::JsonSchemaDialect::Draft7},
        {R"json({"items": {"type": "object", "properties": {"child": true}}})json",
         R"json(["x", "x", {"child": {}}])json"},
        {R"json({"minItems": 3, "maxItems": 1, "contains": {"allOf": [true]}})json", "[1, 2]"},
        {R"json({"minItems": 3, "uniqueItems": true, "contains": {"allOf": [true]}})json",
         "[1, 1]"},
        {R"json({"unevaluatedItems":
                 {"type": "object", "properties": {"child": true}}})json",
         R"json(["x", "x", {"child": {}}])json"},
    };
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto schema = rad::ParseJson(testCase.schema);
        const auto instance = rad::ParseJson(testCase.instance);
        ASSERT_TRUE(schema);
        ASSERT_TRUE(instance);
        const auto compiled =
            rad::JsonSchema::Compile(schema.value(), {.dialect = testCase.dialect});
        ASSERT_TRUE(compiled) << compiled.error().message;
        for (const std::size_t maxErrors : {0U, 1U, 2U, 3U, 64U})
        {
            SCOPED_TRACE(maxErrors);
            rad::JsonSchemaValidationOptions options;
            options.maxErrors = maxErrors;
            options.maxEvaluationDepth = 1;
            const auto result = compiled.value().Validate(instance.value(), options);
            EXPECT_FALSE(result);
            const std::size_t expectedErrors = maxErrors <= 1 ? 1 : maxErrors == 2 ? 2 : 3;
            ASSERT_EQ(result.errors.size(), expectedErrors) << FormatValidationErrors(result);
            for (std::size_t index = 0; index < result.errors.size(); ++index)
            {
                EXPECT_EQ(result.errors[index].errorCode,
                          index == 2 ? rad::JsonSchemaValidationErrorCode::DepthLimitExceeded
                                     : rad::JsonSchemaValidationErrorCode::SchemaViolation)
                    << FormatValidationErrors(result);
            }
        }
    }
}

TEST(IO, JsonSchemaSpeculativeBranchErrorLimits)
{
    struct Case
    {
        std::string_view schema;
        std::string_view instance;
        bool valid;
    };
    constexpr Case cases[] = {
        {R"json({"anyOf": [{"allOf": [false, {"allOf": [true]}]}, true]})json", "1", true},
        {R"json({"oneOf": [{"allOf": [false, {"allOf": [true]}]}, true]})json", "1", true},
        {R"json({"not": {"allOf": [false, {"allOf": [true]}]}})json", "1", true},
        {R"json({"if": {"allOf": [false, {"allOf": [true]}]},
                 "then": false, "else": true})json",
         "1", true},
        {R"json({"contains": {"type": "integer", "properties": {"child": true}}})json",
         R"json([{"child": {}}, 1])json", true},
        {R"json({"anyOf": [{"allOf": [false, {"allOf": [true]}]}, false]})json", "1", false},
    };
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto schema = rad::ParseJson(testCase.schema);
        const auto instance = rad::ParseJson(testCase.instance);
        ASSERT_TRUE(schema);
        ASSERT_TRUE(instance);
        const auto compiled = rad::JsonSchema::Compile(
            schema.value(), {.dialect = rad::JsonSchemaDialect::Draft2020_12});
        ASSERT_TRUE(compiled) << compiled.error().message;
        for (const std::size_t maxErrors : {0U, 1U, 2U})
        {
            rad::JsonSchemaValidationOptions options;
            options.maxErrors = maxErrors;
            options.maxEvaluationDepth = 2;
            const auto result = compiled.value().Validate(instance.value(), options);
            EXPECT_EQ(static_cast<bool>(result), testCase.valid) << FormatValidationErrors(result);
            EXPECT_EQ(result.errors.size(), testCase.valid ? 0 : 1);
            for (const auto& error : result.errors)
            {
                EXPECT_EQ(error.errorCode, rad::JsonSchemaValidationErrorCode::SchemaViolation);
            }
        }
    }
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
        options.maxEvaluationDepth = 0;
        options.maxErrors = 1;
        const auto limited = compiled.value().Validate(rad::JsonArray{1}, options);
        ASSERT_FALSE(limited);
        ASSERT_EQ(limited.errors.size(), 1);
        EXPECT_EQ(limited.errors[0].errorCode,
                  rad::JsonSchemaValidationErrorCode::DepthLimitExceeded);
        EXPECT_EQ(limited.errors[0].instancePath, "/0");
        EXPECT_EQ(limited.errors[0].schemaPath, "/unevaluatedItems");
        EXPECT_EQ(limited.errors[0].message, "maximum validation depth exceeded");

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
