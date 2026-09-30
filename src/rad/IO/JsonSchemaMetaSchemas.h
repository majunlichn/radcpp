#pragma once

#include <rad/IO/Json.h>

#include <optional>
#include <string>
#include <string_view>

namespace rad::detail
{

[[nodiscard]] std::optional<std::string_view> FindJsonSchemaMetaSchema(std::string_view uri);
[[nodiscard]] std::string_view JsonSchemaCoreVocabularyUri(JsonSchemaDialect dialect);
[[nodiscard]] Result<bool, std::string> GetJsonSchemaVocabularySupport(
    std::string_view uri, JsonSchemaDialect dialect);

} // namespace rad::detail
