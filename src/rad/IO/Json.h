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

// Compiles and validates a practical subset of JSON Schema Draft 7, 2019-09, and 2020-12.
// Compiled schemas own their documents and expressions; copies share immutable state.
//
// Supported keywords:
// - All drafts: boolean schemas; $ref, $id, and JSON Pointers; type, enum, const;
//   numeric bounds and multipleOf; min/max string, array, and object sizes; pattern;
//   required, properties, patternProperties, propertyNames, additionalProperties;
//   single-schema items, uniqueItems, contains; allOf, anyOf, oneOf, not, and if/then/else.
// - Draft 7: dependencies (property-name arrays and schemas), definitions structure checking,
//   and plain-name fragments in $id.
// - Draft 7 and 2019-09: tuple-form items and additionalItems.
// - Draft 2019-09 and 2020-12: dependentRequired, dependentSchemas, minContains, maxContains,
//   unevaluatedProperties, unevaluatedItems, $anchor, $defs structure checking, and $vocabulary.
// - Draft 2019-09: $recursiveRef (only "#") and boolean $recursiveAnchor.
// - Draft 2020-12: prefixItems, $dynamicRef, and $dynamicAnchor.
//
// References:
// - Resolution is offline, using the root schema, registered documents, and bundled standard
//   meta-schemas and their default vocabulary meta-schemas. JSON Pointers can address definitions
//   and $defs. Existing resources and registered documents take precedence over bundled ones;
//   a missing fragment in a registered document is an error, not a built-in fallback.
// - $ref targets are static. $dynamicRef can rebind named dynamic anchors through the active
//   resource scope; JSON Pointer targets and ordinary anchors remain static.
// - CompileFile reads only the supplied file and does not use its path as a retrieval URI.
//
// Dialects and vocabularies:
// - Draft 2019-09 and 2020-12 support registered custom meta-schemas. A resource's $schema
//   selects its vocabulary profile and underlying standard draft, reported by Dialect().
// - Registered documents without $schema inherit the root dialect and vocabulary profile.
//   A vocabulary-changing $schema in a subschema requires its own $id resource boundary.
// - The selected meta-schema's $vocabulary enables the listed supported vocabularies, even
//   when optional (false); omitted vocabularies are disabled. An absent $vocabulary uses
//   standard defaults. Unknown optional vocabularies are ignored; unsupported required ones fail.
// - A schema's own $vocabulary does not change its keyword enablement. Compilation checks
//   supported keyword definitions but does not automatically validate against the meta-schema.
//
// Numeric semantics:
// - Instances must contain only finite numbers, including in unconstrained objects and arrays.
//   Non-finite values fail before schema evaluation, even for boolean schemas; diagnostics use
//   the offending instance paths and the root schema location.
//   This check traverses the instance once without applying the schema evaluation depth limit.
// - Enabled enum and const assertions reject non-finite values in their literal payloads.
// - multipleOf uses exact arithmetic on integers and shortest round-trip decimal representations
//   of doubles, without an epsilon. Original numeric text is not retained; a computed value
//   such as 0.1 + 0.2 need not be a multiple of 0.1.
//
// Regular expressions:
// - pattern and patternProperties use PCRE2 with UTF-8 and Unicode category escape support.
//   Expressions compile once per schema location; matching scratch state is local to each call.
// - Matching is unanchored unless the pattern supplies anchors. \d and \w are ASCII by default;
//   \s and \S use ECMAScript whitespace semantics, including inside character classes.
//   Other PCRE2 syntax and semantics are not fully ECMAScript-compatible.
// - PCRE2 limits: match limit 1,000,000, depth 1,000, and 8 MiB of matching heap.
// - For debugging, RAD_JSON_SCHEMA_USE_STD_REGEX=1 selects std::regex ECMAScript over UTF-8
//   bytes instead; this backend is not fully Unicode-aware.
//
// Limitations:
// - No automatic file/network loading, mixed-base-draft resources, custom Draft 7 dialects,
//   custom keyword implementations, or unsupported required vocabularies (including format
//   assertion). format and other annotation keywords do not assert instance validity.
class JsonSchema
{
public:
    // Detects the underlying standard dialect from the required root $schema keyword.
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

    // Returns the underlying standard draft, including for a custom vocabulary dialect.
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
