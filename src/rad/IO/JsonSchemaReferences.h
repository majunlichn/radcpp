#pragma once

#include <rad/IO/Json.h>

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
    using Resolution = Result<std::string, JsonSchemaCompileError>;

    [[nodiscard]] static Result<JsonSchemaReferences, JsonSchemaCompileError> Compile(
        const JsonValue& schema, JsonSchemaDialect dialect, std::size_t maxDepth,
        const JsonSchemaCompileOptions& options);

    [[nodiscard]] const Resolution* Find(std::string_view referencePath) const;
    [[nodiscard]] const JsonValue& Documents() const;
    [[nodiscard]] std::string SchemaPath(std::string_view path) const;
    [[nodiscard]] std::string SchemaUri(std::string_view path) const;
    [[nodiscard]] const std::string* Resource(std::string_view schemaPath) const;
    [[nodiscard]] bool HasRecursiveAnchor(std::string_view resourcePath) const;

private:
    std::unordered_map<std::string, Resolution> m_targets;
    // Internal paths begin with a document index; public diagnostics omit that prefix.
    JsonValue m_documents;
    std::vector<std::string> m_uris;
    std::unordered_map<std::string, std::string> m_schemaResources;
}; // class JsonSchemaReferences

} // namespace rad::detail
