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
        const JsonValue& schema, JsonSchemaDialect dialect, std::size_t maxDepth);

    [[nodiscard]] const Resolution* Find(std::string_view referencePath) const;

private:
    std::unordered_map<std::string, Resolution> m_targets;
}; // class JsonSchemaReferences

} // namespace rad::detail
