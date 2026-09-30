#pragma once

#include <rad/IO/Json.h>

#include "JsonSchemaDialects.h"
#include "JsonSchemaRegex.h"

#include <string>
#include <string_view>
#include <unordered_map>

namespace rad::detail
{

[[nodiscard]] std::string ChildPath(std::string_view path, std::string_view token);
[[nodiscard]] const JsonValue* FindJsonSchemaValue(const JsonValue& root, std::string_view pointer);

class JsonSchemaReferences
{
public:
    struct Target
    {
        std::string path;
        std::string dynamicAnchor;
    };

    using Resolution = Result<Target, JsonSchemaCompileError>;
    using Anchors = std::unordered_map<std::string, std::string>;
    using Patterns = std::unordered_map<std::string, JsonSchemaRegex>;

    [[nodiscard]] static Result<JsonSchemaReferences, JsonSchemaCompileError> Compile(
        const JsonValue& schema, std::optional<JsonSchemaDialect> dialect, std::size_t maxDepth,
        const JsonSchemaCompileOptions& options);

    [[nodiscard]] const Resolution* Find(std::string_view referencePath) const;
    [[nodiscard]] const JsonValue& Documents() const;
    [[nodiscard]] std::string SchemaPath(std::string_view path) const;
    [[nodiscard]] std::string SchemaUri(std::string_view path) const;
    [[nodiscard]] const std::string* Resource(std::string_view schemaPath) const;
    [[nodiscard]] bool HasRecursiveAnchor(std::string_view resourcePath) const;
    [[nodiscard]] const Anchors* DynamicAnchors(std::string_view resourcePath) const;
    [[nodiscard]] JsonSchemaDialect Dialect() const;
    [[nodiscard]] const JsonSchemaVocabularyProfile* Profile(std::string_view schemaPath) const;
    [[nodiscard]] const JsonSchemaCompileError* ProfileError(std::string_view schemaPath) const;
    void SetPatterns(Patterns patterns);
    [[nodiscard]] const JsonSchemaRegex* FindPattern(std::string_view schemaPath) const;

private:
    std::unordered_map<std::string, Resolution> m_targets;
    // Internal paths begin with a document index; public diagnostics omit that prefix.
    JsonValue m_documents;
    std::vector<std::string> m_uris;
    std::unordered_map<std::string, std::string> m_schemaResources;
    std::unordered_map<std::string, Anchors> m_dynamicAnchors;
    JsonSchemaDialect m_dialect = JsonSchemaDialect::Draft2020_12;
    std::unordered_map<std::string, JsonSchemaVocabularyProfile> m_profiles;
    std::unordered_map<std::string, JsonSchemaCompileError> m_profileErrors;
    Patterns m_patterns;
}; // class JsonSchemaReferences

} // namespace rad::detail
