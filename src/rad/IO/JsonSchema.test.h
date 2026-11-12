#pragma once

#include <rad/IO/JsonSchema.h>

#include <format>
#include <string>
#include <utility>

namespace rad::test
{

[[nodiscard]] inline Ref<JsonDocument> MakeJsonDocument(JsonValue root, std::string uri = {})
{
    auto document = JsonDocument::FromValue(std::move(root));
    document->SetUri(std::move(uri));
    return document;
}

[[nodiscard]] inline std::string
FormatValidationErrors(const rad::JsonSchemaValidationResult& result)
{
    std::string output = std::format("status={}\n", static_cast<int>(result.status));
    for (const auto& error : result.errors)
    {
        output += std::format("instance={}, document={}, schema={}: {}\n", error.instancePath,
                              error.schemaUri, error.schemaPath, error.message);
    }
    return output;
}

} // namespace rad::test
