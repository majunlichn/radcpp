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
        output += std::format("code={}, instance={}, document={}, schema={}: {}\n",
                              static_cast<int>(error.errorCode), error.instancePath,
                              error.schemaUri, error.schemaPath, error.message);
    }
    return output;
}

} // namespace rad::test
