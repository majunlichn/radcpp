#pragma once

#include <rad/IO/Json.h>

namespace rad::detail
{

enum class JsonSchemaVocabularyErrorKind
{
    NotObject,
    MissingCore,
    InvalidCore,
    InvalidRequirement,
    InvalidUri,
    UnsupportedRequired,
};

struct JsonSchemaVocabularyError
{
    JsonSchemaVocabularyErrorKind kind;
    std::string path;
    std::string message;
};

[[nodiscard]] inline std::exception_ptr make_exception_ptr(const JsonSchemaVocabularyError& error)
{
    return std::make_exception_ptr(error);
}

struct JsonSchemaVocabularyDeclaration
{
    bool applicator = false;
    bool validation = false;
    bool unevaluated = false;
};

[[nodiscard]] Result<JsonSchemaVocabularyDeclaration, JsonSchemaVocabularyError>
ParseJsonSchemaVocabularies(const JsonValue& value, JsonSchemaDialect dialect);

} // namespace rad::detail
