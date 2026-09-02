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
    // A zero value is treated as one.
    std::size_t maxErrors = 64;
    std::size_t maxDepth = 128;
};

// Implements a practical subset of JSON Schema Draft 7, Draft 2019-09, and Draft 2020-12.
//
// Supported:
// - All dialects: boolean schemas; type, enum, const; numeric bounds and integer multipleOf;
//   min/max string, array, and object sizes; pattern; required, properties,
//   additionalProperties; single-schema items, uniqueItems, contains; allOf, anyOf, oneOf,
//   not; and if/then/else.
// - Draft 2019-09 and 2020-12: dependentRequired, dependentSchemas, minContains, maxContains,
//   and $defs structure checking.
// - Draft 2020-12: prefixItems.
//
// Not supported:
// - $ref or anchor resolution, vocabularies, patternProperties, propertyNames,
//   unevaluatedProperties, unevaluatedItems, and fractional multipleOf.
// - Draft 7: dependencies.
// - Draft 7 and 2019-09: tuple-form items and additionalItems.
//
// definitions and $defs are checked but cannot be referenced.
// Identification and annotation keywords are ignored; format is not validated.
// pattern uses std::regex ECMAScript syntax over UTF-8 bytes and is not fully Unicode-aware.
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
