#pragma once

#include <rad/Core/Result.h>
#include <rad/System/FileSystem.h>

#include <boost/json/array.hpp>
#include <boost/json/kind.hpp>
#include <boost/json/object.hpp>
#include <boost/json/parse_options.hpp>
#include <boost/json/string.hpp>
#include <boost/json/value.hpp>
#include <boost/system/error_code.hpp>

#include <cstddef>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rad
{

using JsonValue = boost::json::value;
using JsonObject = boost::json::object;
using JsonArray = boost::json::array;
using JsonString = boost::json::string;
using JsonKind = boost::json::kind;
using JsonErrorCode = boost::system::error_code;
using JsonParseOptions = boost::json::parse_options;

[[nodiscard]] Result<JsonValue, JsonErrorCode> ParseJson(std::string_view text);
[[nodiscard]] Result<JsonValue, JsonErrorCode> ParseJson(std::string_view text,
                                                         const JsonParseOptions& options);

[[nodiscard]] std::string PrettyJson(const JsonValue& value, std::string_view indent = "  ");

enum class JsonSchemaDialect
{
    Draft7,
    Draft2019_09,
    Draft2020_12,
};

enum class JsonSchemaCompileErrorCode
{
    MissingDialect,
    UnsupportedDialect,
    FileReadError,
    InvalidJson,
    UnsupportedFeature,
    InvalidSchema,
};

struct JsonSchemaCompileError
{
    JsonSchemaCompileErrorCode code;
    std::optional<JsonSchemaDialect> dialect;
    std::string schemaPath;
    std::string message;
};

[[nodiscard]] inline std::exception_ptr
make_exception_ptr(const JsonSchemaCompileError& error)
{
    return std::make_exception_ptr(error);
}

struct JsonSchemaValidationError
{
    std::string instancePath;
    std::string schemaPath;
    std::string message;
};

struct JsonSchemaValidationResult
{
    std::vector<JsonSchemaValidationError> errors;

    [[nodiscard]] explicit operator bool() const noexcept { return errors.empty(); }
};

struct JsonSchemaValidationOptions
{
    // Limits diagnostics, not evaluation. A zero value is treated as one.
    std::size_t maxErrors = 64;
    std::size_t maxDepth = 128;
};

// Implements a practical subset of JSON Schema Draft 7, Draft 2019-09, and Draft 2020-12.
//
// Supported:
// - All dialects: boolean schemas; root-local JSON Pointer $ref; type, enum, const;
//   numeric bounds and integer multipleOf;
//   min/max string, array, and object sizes; pattern; required, properties,
//   patternProperties, propertyNames,
//   additionalProperties; single-schema items, uniqueItems, contains; allOf, anyOf, oneOf,
//   not; and if/then/else.
// - Draft 2019-09 and 2020-12: dependentRequired, dependentSchemas, minContains, maxContains,
//   unevaluatedProperties, and $defs structure checking.
// - Draft 2020-12: prefixItems.
// - Draft 7: dependencies (property-name arrays and schemas).
// - Draft 7 and 2019-09: tuple-form items and additionalItems.
//
// Not supported:
// - Remote/relative references, anchors, embedded $id resources, vocabularies,
//   unevaluatedItems and fractional multipleOf.
//
// definitions and $defs can be referenced by root-local JSON Pointers.
// Embedded $id resources cannot be used as reference sources or targets.
// Identification and annotation keywords are otherwise ignored; format is not validated.
// pattern and patternProperties use PCRE2 with UTF-8 and Unicode category escape support.
// Matching is unanchored unless the pattern supplies anchors; \d and \w are ASCII by default.
// ECMAScript whitespace semantics are used for \s and \S, including inside character classes.
// Other PCRE2 syntax and semantics are not fully ECMAScript-compatible.
// PCRE2 matching limits: match limit 1,000,000, depth 1,000, and 8 MiB of matching heap.
// For debugging, rebuild rad with RAD_JSON_SCHEMA_USE_STD_REGEX=1 to use std::regex
// ECMAScript over UTF-8 bytes instead; that backend is not fully Unicode-aware.
class JsonSchema
{
public:
    // Detects the dialect from the required root $schema keyword.
    [[nodiscard]] static Result<JsonSchema, JsonSchemaCompileError>
    Compile(const JsonValue& schema);
    [[nodiscard]] static Result<JsonSchema, JsonSchemaCompileError>
    Compile(const JsonValue& schema, JsonSchemaDialect dialect);
    [[nodiscard]] static Result<JsonSchema, JsonSchemaCompileError>
    CompileFile(const FilePath& path, JsonSchemaDialect dialect);

    [[nodiscard]] JsonSchemaDialect Dialect() const noexcept;
    [[nodiscard]] JsonSchemaValidationResult
    Validate(const JsonValue& instance,
             const JsonSchemaValidationOptions& options = {}) const;

private:
    JsonSchema(JsonValue schema, JsonSchemaDialect dialect);

    JsonValue m_schema;
    JsonSchemaDialect m_dialect;
}; // class JsonSchema

} // namespace rad
