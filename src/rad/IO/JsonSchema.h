#pragma once

#include <rad/IO/Json.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace rad
{
namespace detail
{
class JsonSchemaReferences;
} // namespace detail

enum class JsonSchemaDialect
{
    Draft7,
    Draft2019_09,
    Draft2020_12,
};

enum class JsonSchemaCompileErrorCode
{
    // Bundled meta-schema text is not valid JSON.
    InvalidJson,

    // The dialect option is not a valid JsonSchemaDialect enumerator.
    InvalidDialect,

    // Schema structure, keyword values, identifiers, or references are invalid.
    InvalidSchema,
    // Required schema semantics are unsupported, or a referenced document is unavailable offline.
    UnsupportedFeature,
    // A schema or meta-schema compilation traversal exceeded maxCompilationDepth.
    DepthLimitExceeded,
};

struct JsonSchemaCompileError
{
    JsonSchemaCompileErrorCode code;
    // Retrieval URI of the schema document, empty for an anonymous root.
    std::string schemaUri;
    // JSON Pointer within the schema document; empty denotes its root.
    std::string schemaPath;
    // Human-readable explanation, not a stable error code.
    std::string message;
};

[[nodiscard]] inline std::exception_ptr
make_exception_ptr(const JsonSchemaCompileError& error)
{
    return std::make_exception_ptr(error);
}

enum class JsonSchemaValidationStatus
{
    // Validation finished without reaching a limit or encountering an evaluation failure.
    // Completion does not imply that the instance conforms to the schema.
    Completed,
    // The effective maxErrors limit was reached; more violations may or may not exist.
    MaxErrorsReached,
    DepthLimitExceeded,
    // An active schema/instance evaluation recurred with equivalent scope and error policy.
    EvaluationCycleDetected,
    RegexResourceLimitExceeded,
    RegexEvaluationFailed,
    InternalError,
};

struct JsonSchemaValidationError
{
    // JSON Pointer into the instance; empty denotes its root. Missing-property locations
    // need not exist.
    std::string instancePath;
    // Retrieval URI of the containing schema document, not its $id; empty for an anonymous root.
    std::string schemaUri;
    // JSON Pointer within that schema document, e.g. "/properties/name/type".
    // Empty denotes the document root, not necessarily the resource root selected by $id.
    std::string schemaPath;
    // Human-readable explanation, not a stable error code.
    std::string message;
};

struct JsonSchemaValidationResult
{
    JsonSchemaValidationStatus status = JsonSchemaValidationStatus::Completed;
    // Completed and MaxErrorsReached contain only schema violations. For any other status,
    // the final entry describes the evaluation failure; preceding entries are schema violations.
    std::vector<JsonSchemaValidationError> errors;

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return status == JsonSchemaValidationStatus::Completed && errors.empty();
    }
};

struct JsonSchemaRegexLimits
{
    // Per PCRE2 match, not per validation call. Zero is restrictive, not unlimited.
    std::uint32_t matchLimit = UINT32_MAX;
    std::uint32_t backtrackingDepthLimit = UINT32_MAX;
    std::uint32_t heapLimitKiB = 4 * 1024 * 1024;
};

struct JsonSchemaValidationOptions
{
    // Stops evaluation once this many schema violations are collected. Zero is treated as one.
    // Reaching the limit reports MaxErrorsReached, even if no further violations exist.
    // Evaluation failures stop immediately and also fit within this effective error limit,
    // but retain their specific failure status even when they fill the final error slot.
    std::size_t maxErrors = 64;
    // Maximum nested schema evaluation depth, not the instance's JSON nesting depth.
    // The root is depth zero; each subschema or reference evaluation adds one.
    // Sibling evaluations do not accumulate depth. Zero allows only root evaluation.
    // Exceeding the limit reports a DepthLimitExceeded status.
    // The default guards recursive evaluation; this is not a total work budget.
    std::size_t maxEvaluationDepth = 128;
    JsonSchemaRegexLimits regex;
};

struct JsonSchemaCompileOptions
{
    // Standard default used when the root schema or a selected meta-schema has no $schema.
    // A root $schema declaration takes precedence, including when its underlying draft differs.
    // Invalid or unsupported declarations report an error instead of using this default.
    // The default is fixed at Draft2020_12, not the latest supported draft.
    // Root $schema selection precedes draft-specific keyword processing, including $ref.
    JsonSchemaDialect dialect = JsonSchemaDialect::Draft2020_12;
    // Documents registered for offline reference and meta-schema resolution.
    // Each must be non-null with an absolute, fragment-free URI.
    // URIs and resource identifiers must not conflict with the root or each other.
    // Unused schemas are not validated.
    std::vector<Ref<const JsonDocument>> documents;
    // Maximum recursive traversal depth during schema and meta-schema compilation.
    // Each traversal starts at zero. Zero is restrictive, not unlimited.
    // Independent of JSON parsing and instance evaluation depth; not a total work budget.
    // Excessive values can exhaust the call stack of the recursive implementation.
    std::size_t maxCompilationDepth = 128;
};

// Compiles and validates a practical subset of JSON Schema Draft 7, 2019-09, and 2020-12.
// Compilation checks definitions, binds static reference targets, and prepares regular expressions.
// JsonSchemaCompileOptions::maxCompilationDepth guards recursive compilation independently of
// parsing and validation options; it is not a JSON nesting or reference-count limit.
// Compiled schemas retain input documents without copying JSON and own prepared expressions.
// Retained documents and their URIs must not be modified while a compiled schema uses them.
// Copies share compiled state.
// Instance evaluation keeps diagnostics and resource scopes local to each call.
// Validation passes only with Completed status and an empty error list.
// Status describes why validation ended; errors contain source locations and explanations.
// uniqueItems reports a repeated element's location; selection among multiple duplicates is
// unspecified and may depend on the input size.
// Value equality for enum, const, and duplicate checks uses bounded recursion with an iterative
// fallback, independently of the schema evaluation depth limit.
// Large-array duplicate checks may reserve O(n) auxiliary storage up front, even when a
// duplicate is found early.
// Validation stops at the effective error limit or the first evaluation failure. Speculative
// branches keep their violations local, so a rejected branch does not reject a valid alternative.
// Reentering an active schema/instance evaluation with equivalent reference scope and error policy
// reports EvaluationCycleDetected, not a violation. Completed sibling evaluations are independent.
// The depth limit still applies before entering an evaluation, including a cyclic one.
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
// - Resolution performs no file or network access; URIs identify registered resources.
//
// Dialects and vocabularies:
// - Draft 2019-09 and 2020-12 support registered custom meta-schemas. A resource's $schema
//   selects its vocabulary profile and underlying standard draft, reported by Dialect().
// - Registered documents without $schema inherit the root draft and vocabulary profile.
//   In Draft 2019-09 and 2020-12, $schema is allowed only at document roots or embedded
//   resource roots identified by $id, even when it declares the same dialect.
// - The selected meta-schema's $vocabulary enables the listed supported vocabularies, even
//   when optional (false); omitted vocabularies are disabled. An absent $vocabulary uses
//   standard defaults. Unknown optional vocabularies are ignored; unsupported required ones fail.
// - A schema's own $vocabulary does not change its keyword enablement. Compilation checks
//   supported keyword definitions but does not automatically validate against the meta-schema.
// - Draft 7 preserves its standard meta-schema's non-empty, unique enum requirement.
//   Draft 2019-09 and 2020-12 allow empty enums and repeated values; empty enums reject all instances.
//
// Numeric semantics:
// - Instances must contain only finite numbers, including in unconstrained objects and arrays.
//   Non-finite values fail before schema evaluation, even for boolean schemas; diagnostics use
//   the offending instance paths and the root schema location.
//   This check traverses the instance once without applying the schema evaluation depth limit.
// - Enabled enum and const assertions reject non-finite values in their literal payloads.
// - Size and count bounds accept finite non-negative integers beyond container size limits.
//   Comparisons do not narrow these bounds to the container's size type.
// - multipleOf uses exact arithmetic on integers and shortest round-trip decimal representations
//   of doubles, without an epsilon. Original numeric text is not retained; a computed value
//   such as 0.1 + 0.2 need not be a multiple of 0.1.
//
// Regular expressions:
// - pattern and patternProperties use PCRE2 with UTF-8 and Unicode category escape support.
//   Expressions compile once per schema location. Matching workspace is initialized lazily and
//   reused within one validation call; concurrent calls keep separate workspace.
// - Matching is unanchored unless the pattern supplies anchors. \d and \w are ASCII by default;
//   \s and \S use ECMAScript whitespace semantics, including inside character classes.
//   Other PCRE2 syntax and semantics are not fully ECMAScript-compatible.
// - JsonSchemaValidationOptions::regex controls each match's limits. Defaults: match limit
//   UINT32_MAX, backtracking depth UINT32_MAX, and 4 GiB of backtracking heap. These are upper
//   bounds, not preallocated resources; lower them for constrained workloads. Exhaustion fails
//   validation with RegexResourceLimitExceeded, not an ordinary pattern mismatch. Other matching
//   failures report RegexEvaluationFailed. Evaluation failures are the final entry in errors,
//   following any schema violations collected before validation stopped.
//
// Limitations:
// - No automatic file/network loading, mixed-base-draft resources, custom Draft 7 dialects,
//   custom keyword implementations, or unsupported required vocabularies (including format
//   assertion). format and other annotation keywords do not assert instance validity.
class JsonSchema
{
public:
    // document must be non-null; URI must be empty or absolute and fragment-free.
    // URI is the initial base before $id; empty means an anonymous root.
    // Null documents and invalid URIs report InvalidSchema.
    [[nodiscard]] static Result<JsonSchema, JsonSchemaCompileError> Compile(
        Ref<const JsonDocument> document, const JsonSchemaCompileOptions& options = {});

    // Returns the root's resolved standard draft. Custom dialects with different
    // vocabularies can share the same draft; this does not identify the full dialect.
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
