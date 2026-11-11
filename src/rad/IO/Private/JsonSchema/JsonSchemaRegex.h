#pragma once

#include <rad/IO/JsonSchema.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <string_view>

struct pcre2_real_code_8;
struct pcre2_real_match_context_8;
struct pcre2_real_match_data_8;

namespace rad::detail
{

struct JsonSchemaRegexError
{
    std::string message;
    JsonSchemaValidationStatus status = JsonSchemaValidationStatus::RegexEvaluationFailed;
};

[[nodiscard]] inline std::exception_ptr make_exception_ptr(const JsonSchemaRegexError& error)
{
    return std::make_exception_ptr(error);
}

class JsonSchemaRegex
{
public:
    class MatchContext
    {
    public:
        explicit MatchContext(const JsonSchemaRegexLimits& limits);
        MatchContext(const MatchContext&) = delete;
        MatchContext& operator=(const MatchContext&) = delete;
        ~MatchContext();

    private:
        friend class JsonSchemaRegex;
        struct ContextDeleter
        {
            void operator()(pcre2_real_match_context_8* context) const noexcept;
        };
        struct DataDeleter
        {
            void operator()(pcre2_real_match_data_8* data) const noexcept;
        };
        using ContextHandle = std::unique_ptr<pcre2_real_match_context_8, ContextDeleter>;
        using DataHandle = std::unique_ptr<pcre2_real_match_data_8, DataDeleter>;

        JsonSchemaRegexLimits m_limits;
        ContextHandle m_context;
        DataHandle m_data;
        std::uint32_t m_capacity = 0;
    }; // class MatchContext

    JsonSchemaRegex(JsonSchemaRegex&&) noexcept;
    JsonSchemaRegex& operator=(JsonSchemaRegex&&) noexcept;
    ~JsonSchemaRegex();

    [[nodiscard]] static Result<JsonSchemaRegex, JsonSchemaRegexError>
    Compile(std::string_view pattern);
    [[nodiscard]] Result<bool, JsonSchemaRegexError> Matches(
        std::string_view value, MatchContext& context) const;

private:
    struct CodeDeleter
    {
        void operator()(pcre2_real_code_8* expression) const noexcept;
    };
    using Expression = std::unique_ptr<pcre2_real_code_8, CodeDeleter>;

    JsonSchemaRegex(Expression expression, std::uint32_t matchPairs);

    Expression m_expression;
    std::uint32_t m_matchPairs;
}; // class JsonSchemaRegex

} // namespace rad::detail
