#pragma once

#include <rad/IO/Json.h>

#include <memory>
#include <optional>
#include <string_view>

namespace rad::detail
{

struct JsonSchemaVocabularyProfile
{
    JsonSchemaDialect dialect;
    bool applicator = true;
    bool validation = true;
    bool unevaluated = true;

    [[nodiscard]] bool IsKeywordEnabled(std::string_view keyword) const;
};

// Root and caller documents must outlive the resolver. Resolution is entirely offline.
class JsonSchemaDialectResolver
{
public:
    JsonSchemaDialectResolver(const JsonValue& root, const JsonSchemaCompileOptions& options,
                              std::optional<JsonSchemaDialect> fallbackDialect,
                              std::size_t maxDepth);
    ~JsonSchemaDialectResolver();

    [[nodiscard]] Result<JsonSchemaVocabularyProfile, JsonSchemaCompileError>
    Resolve(std::string_view uri);
    // Source coordinates are public document URIs and JSON Pointers, not internal paths.
    [[nodiscard]] Result<JsonSchemaVocabularyProfile, JsonSchemaCompileError>
    Resolve(std::string_view uri, std::string_view sourceUri, std::string_view sourcePath);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
}; // class JsonSchemaDialectResolver

} // namespace rad::detail
