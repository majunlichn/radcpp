#pragma once

#include <rad/IO/Json.h>

#include <format>
#include <string>

namespace rad::test
{

[[nodiscard]] inline std::string
FormatValidationErrors(const rad::JsonSchemaValidationResult& result)
{
    std::string output;
    for (const auto& error : result.errors)
    {
        output += std::format("instance={}, document={}, schema={}: {}\n", error.instancePath,
                              error.schemaUri, error.schemaPath, error.message);
    }
    if (result.evaluationError)
    {
        const auto& error = *result.evaluationError;
        output += std::format("evaluation status={}, instance={}, document={}, schema={}: {}\n",
                              static_cast<int>(result.status), error.instancePath, error.schemaUri,
                              error.schemaPath, error.message);
    }
    return output;
}

} // namespace rad::test
