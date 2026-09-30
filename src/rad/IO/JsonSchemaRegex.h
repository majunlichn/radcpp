#pragma once

#include <rad/IO/Json.h>

#include <exception>
#include <memory>
#include <string>
#include <string_view>

namespace rad::detail
{

struct JsonSchemaRegexError
{
    bool unsupported;
    bool resource;
    std::string message;
};

[[nodiscard]] inline std::exception_ptr make_exception_ptr(const JsonSchemaRegexError& error)
{
    return std::make_exception_ptr(error);
}

class JsonSchemaRegex
{
public:
    JsonSchemaRegex(JsonSchemaRegex&&) noexcept;
    JsonSchemaRegex& operator=(JsonSchemaRegex&&) noexcept;
    ~JsonSchemaRegex();

    [[nodiscard]] static Result<JsonSchemaRegex, JsonSchemaRegexError>
    Compile(std::string_view pattern);
    [[nodiscard]] Result<bool, JsonSchemaRegexError> Matches(
        std::string_view value, const JsonSchemaRegexMatchLimits& limits) const;

private:
    struct Impl;
    explicit JsonSchemaRegex(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> m_impl;
}; // class JsonSchemaRegex

} // namespace rad::detail
