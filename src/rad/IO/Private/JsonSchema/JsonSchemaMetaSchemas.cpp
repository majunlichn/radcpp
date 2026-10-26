#include "JsonSchemaMetaSchemas.h"
#include "JsonSchemaMetaSchemasData.h"

#include <boost/url.hpp>

#include <algorithm>
#include <array>

namespace rad::detail
{

std::optional<std::string_view> FindJsonSchemaMetaSchema(std::string_view uri)
{
    for (const auto& [identifier, text] : JsonSchemaMetaSchemaTexts)
    {
        if (identifier == uri)
        {
            return text;
        }
    }
    return std::nullopt;
}

std::string_view JsonSchemaCoreVocabularyUri(JsonSchemaDialect dialect)
{
    switch (dialect)
    {
    case JsonSchemaDialect::Draft2019_09:
        return "https://json-schema.org/draft/2019-09/vocab/core";
    case JsonSchemaDialect::Draft2020_12:
        return "https://json-schema.org/draft/2020-12/vocab/core";
    case JsonSchemaDialect::Auto:
    case JsonSchemaDialect::Draft7:
        return {};
    }
    return {};
}

Result<bool, std::string> GetJsonSchemaVocabularySupport(
    std::string_view uri, JsonSchemaDialect dialect)
{
    const auto parsed = boost::urls::parse_uri(uri);
    if (!parsed)
    {
        return Failure("vocabulary name must be an absolute URI: " + parsed.error().message());
    }
    boost::urls::url normalized(*parsed);
    normalized.normalize();
    if (normalized.buffer() != uri)
    {
        return Failure(std::string("vocabulary URI must be normalized"));
    }
    const auto core = JsonSchemaCoreVocabularyUri(dialect);
    if (core.empty())
    {
        return Success(false);
    }
    const auto prefix = core.substr(0, core.size() - std::string_view("core").size());
    if (!uri.starts_with(prefix))
    {
        return Success(false);
    }
    const auto name = uri.substr(prefix.size());
    constexpr std::array<std::string_view, 5> supported = {
        "core", "applicator", "validation", "meta-data", "content",
    };
    return Success(std::find(supported.begin(), supported.end(), name) != supported.end() ||
                   (dialect == JsonSchemaDialect::Draft2020_12 &&
                    (name == "unevaluated" || name == "format-annotation")));
}

} // namespace rad::detail
