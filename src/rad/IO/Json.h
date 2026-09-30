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
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rad
{
namespace detail
{
class JsonSchemaReferences;
} // namespace detail

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
    // Retrieval URI of the schema document, empty for an anonymous root.
    std::string schemaUri;
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
    std::string schemaUri;
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

struct JsonSchemaDocument
{
    // Absolute, fragment-free retrieval URI. No file or network access is performed.
    std::string uri;
    JsonValue schema;
};

struct JsonSchemaCompileOptions
{
    // Empty means an anonymous root; otherwise an absolute, fragment-free retrieval URI.
    std::string retrievalUri;
    // Compilation owns copies of reachable documents; unused documents are not validated.
    // Retrieval URIs and resource identifiers must not conflict.
    std::vector<JsonSchemaDocument> documents;
};

// Implements a practical subset of JSON Schema Draft 7, Draft 2019-09, and Draft 2020-12.
//
// Supported:
// - All dialects: boolean schemas; $ref across registered documents, $id resources,
//   and JSON Pointers;
//   type, enum, const;
//   numeric bounds and multipleOf;
//   min/max string, array, and object sizes; pattern; required, properties,
//   patternProperties, propertyNames,
//   additionalProperties; single-schema items, uniqueItems, contains; allOf, anyOf, oneOf,
//   not; and if/then/else.
// - Draft 2019-09 and 2020-12: dependentRequired, dependentSchemas, minContains, maxContains,
//   unevaluatedProperties, unevaluatedItems, $anchor, and $defs structure checking.
// - Offline references to standard meta-schemas and their default vocabulary meta-schemas.
// - Standard $vocabulary declarations; unsupported optional vocabularies are ignored.
// - Draft 2020-12: prefixItems, $dynamicRef, and $dynamicAnchor.
// - Draft 2019-09: $recursiveRef (only "#") and boolean $recursiveAnchor.
// - Draft 7: dependencies (property-name arrays and schemas).
// - Draft 7 and 2019-09: tuple-form items and additionalItems.
// - Draft 7: plain-name fragments in $id.
//
// Not supported:
// - Automatic file/network loading, mixed-dialect resources, custom meta-schema dialects,
//   and unsupported required vocabularies (including format assertion).
//
// definitions and $defs can be referenced by root-local JSON Pointers.
// References resolve within the root schema, caller-provided registry, and bundled meta-schemas.
// Existing resources and caller-provided documents take precedence over bundled meta-schemas.
// A missing fragment in a caller-provided document is an error, not a built-in fallback.
// Registered documents without $schema use the selected dialect.
// Draft 2020-12 $dynamicRef rebinds named dynamic anchors through the active resource scope.
// JSON Pointer targets and ordinary anchors remain static, as do all $ref targets.
// CompileFile does not load referenced files or use its path as a retrieval URI.
// Compilation does not automatically validate schemas against their meta-schema.
// $vocabulary declarations do not change keyword enablement under the selected standard dialect.
// Other annotation keywords are ignored; format is not validated.
// multipleOf uses exact integer arithmetic for integer values and shortest round-trip decimal
// representations for doubles, without an epsilon. Original numeric text is not retained;
// a computed value such as 0.1 + 0.2 need not be a multiple of 0.1.
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
    Compile(const JsonValue& schema, const JsonSchemaCompileOptions& options);
    [[nodiscard]] static Result<JsonSchema, JsonSchemaCompileError>
    Compile(const JsonValue& schema, JsonSchemaDialect dialect);
    [[nodiscard]] static Result<JsonSchema, JsonSchemaCompileError>
    Compile(const JsonValue& schema, JsonSchemaDialect dialect,
            const JsonSchemaCompileOptions& options);
    [[nodiscard]] static Result<JsonSchema, JsonSchemaCompileError>
    CompileFile(const FilePath& path, JsonSchemaDialect dialect);
    [[nodiscard]] static Result<JsonSchema, JsonSchemaCompileError>
    CompileFile(const FilePath& path, JsonSchemaDialect dialect,
                const JsonSchemaCompileOptions& options);

    [[nodiscard]] JsonSchemaDialect Dialect() const noexcept;
    [[nodiscard]] JsonSchemaValidationResult
    Validate(const JsonValue& instance,
             const JsonSchemaValidationOptions& options = {}) const;

private:
    JsonSchema(JsonSchemaDialect dialect,
               std::shared_ptr<const detail::JsonSchemaReferences> references);

    JsonSchemaDialect m_dialect;
    std::shared_ptr<const detail::JsonSchemaReferences> m_references;
}; // class JsonSchema

} // namespace rad
