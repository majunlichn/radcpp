#include "JsonSchemaVocabularies.h"
#include "JsonSchemaMetaSchemas.h"
#include "JsonSchemaPaths.h"

namespace rad::detail
{

Result<JsonSchemaVocabularyDeclaration, JsonSchemaVocabularyError> ParseJsonSchemaVocabularies(
    const JsonValue& value, JsonSchemaDialect dialect)
{
    using Kind = JsonSchemaVocabularyErrorKind;
    if (!value.is_object())
    {
        return Failure(
            JsonSchemaVocabularyError{Kind::NotObject, {}, "$vocabulary must be an object"});
    }
    const auto core = JsonSchemaCoreVocabularyUri(dialect);
    const auto* requiredCore = value.as_object().if_contains(core);
    if (core.empty() || requiredCore == nullptr)
    {
        return Failure(JsonSchemaVocabularyError{
            Kind::MissingCore, {}, "$vocabulary must require the core vocabulary"});
    }
    if (!requiredCore->is_bool() || !requiredCore->as_bool())
    {
        return Failure(JsonSchemaVocabularyError{Kind::InvalidCore, ChildPath("", core),
                                                 "core vocabulary must be required (true)"});
    }
    JsonSchemaVocabularyDeclaration declaration;
    const auto prefix = core.substr(0, core.size() - std::string_view("core").size());
    for (const auto& entry : value.as_object())
    {
        const std::string_view uri = entry.key();
        const auto path = ChildPath("", uri);
        if (!entry.value().is_bool())
        {
            return Failure(JsonSchemaVocabularyError{Kind::InvalidRequirement, path,
                                                     "vocabulary requirement must be a boolean"});
        }
        const auto supported = GetJsonSchemaVocabularySupport(uri, dialect);
        if (!supported)
        {
            return Failure(JsonSchemaVocabularyError{Kind::InvalidUri, path, supported.error()});
        }
        if (!supported.value() && entry.value().as_bool())
        {
            return Failure(
                JsonSchemaVocabularyError{Kind::UnsupportedRequired, path, std::string(uri)});
        }
        if (supported.value() && uri.starts_with(prefix))
        {
            const auto name = uri.substr(prefix.size());
            declaration.applicator |= name == "applicator";
            declaration.validation |= name == "validation";
            declaration.unevaluated |= name == "unevaluated";
        }
    }
    if (dialect == JsonSchemaDialect::Draft2019_09)
    {
        declaration.unevaluated = declaration.applicator;
    }
    return Success(declaration);
}

} // namespace rad::detail
