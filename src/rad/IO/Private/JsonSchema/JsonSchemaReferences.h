#pragma once

#include <rad/IO/JsonSchema.h>

#include "JsonSchemaDialects.h"
#include "JsonSchemaRegex.h"
#include "JsonSchemaPaths.h"

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace rad::detail
{

struct JsonSchemaStringHash
{
    using is_transparent = void;

    [[nodiscard]] std::size_t operator()(std::string_view value) const noexcept
    {
        return std::hash<std::string_view>{}(value);
    }
};

template <typename Value>
using JsonSchemaStringMap =
    std::unordered_map<std::string, Value, JsonSchemaStringHash, std::equal_to<>>;

class JsonSchemaReferences
{
public:
    JsonSchemaReferences() = default;
    JsonSchemaReferences(const JsonSchemaReferences&) = delete;
    JsonSchemaReferences& operator=(const JsonSchemaReferences&) = delete;
    JsonSchemaReferences(JsonSchemaReferences&&) = default;
    JsonSchemaReferences& operator=(JsonSchemaReferences&&) = default;

    struct Target
    {
        std::string path;
        std::string dynamicAnchor;
        // Points into a retained document after reference discovery finishes.
        const JsonValue* schema = nullptr;
    };

    using Resolution = Result<Target, JsonSchemaCompileError>;
    using Anchors = std::unordered_map<std::string, std::string>;
    using Patterns = JsonSchemaStringMap<JsonSchemaRegex>;

    [[nodiscard]] static Result<JsonSchemaReferences, JsonSchemaCompileError> Compile(
        const Ref<const JsonDocument>& document, std::optional<JsonSchemaDialect> dialect,
        std::size_t maxDepth, const JsonSchemaCompileOptions& options);

    [[nodiscard]] const Resolution* Find(std::string_view referencePath) const;
    [[nodiscard]] const std::vector<Ref<const JsonDocument>>& Documents() const;
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
    JsonSchemaStringMap<Resolution> m_targets;
    // Internal paths begin with a document index; public diagnostics omit that prefix.
    std::vector<Ref<const JsonDocument>> m_documents;
    std::vector<std::string> m_uris;
    JsonSchemaStringMap<std::string> m_schemaResources;
    JsonSchemaStringMap<Anchors> m_dynamicAnchors;
    JsonSchemaDialect m_dialect = JsonSchemaDialect::Draft2020_12;
    JsonSchemaStringMap<JsonSchemaVocabularyProfile> m_profiles;
    JsonSchemaStringMap<JsonSchemaCompileError> m_profileErrors;
    Patterns m_patterns;
}; // class JsonSchemaReferences

} // namespace rad::detail
