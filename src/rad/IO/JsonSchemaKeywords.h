#pragma once

#include "JsonSchemaPaths.h"

#include <array>
#include <span>

namespace rad::detail
{

enum class JsonSchemaKeywordVocabulary
{
    Core,
    Applicator,
    Validation,
    Unevaluated,
};

enum class JsonSchemaChildShape
{
    Unsupported,
    None,
    Schema,
    Array,
    Map,
    Dependencies,
    SchemaOrArray,
};

struct JsonSchemaKeyword
{
    std::string_view name;
    JsonSchemaKeywordVocabulary vocabulary;
    // Draft 7, 2019-09, and 2020-12, respectively.
    std::array<JsonSchemaChildShape, 3> shapes;
    bool retainedBesideDraft7Ref = false;

    [[nodiscard]] JsonSchemaChildShape Shape(JsonSchemaDialect dialect) const;
    // Candidate discovery precedes dialect resolution and needs the union of supported shapes.
    [[nodiscard]] JsonSchemaChildShape CandidateShape() const;
    [[nodiscard]] bool AcceptsChild(JsonSchemaDialect dialect, bool arrayChild) const;
};

[[nodiscard]] const JsonSchemaKeyword* FindJsonSchemaKeyword(std::string_view name);
[[nodiscard]] std::span<const JsonSchemaKeyword> JsonSchemaChildKeywords();

template <typename Visitor>
void ForEachJsonSchemaChild(const JsonValue& value, std::string_view path,
                            JsonSchemaChildShape shape, Visitor visitor)
{
    if (shape == JsonSchemaChildShape::SchemaOrArray)
    {
        shape = value.is_array() ? JsonSchemaChildShape::Array : JsonSchemaChildShape::Schema;
    }
    if (shape == JsonSchemaChildShape::Schema)
    {
        visitor(value, std::string(path), false);
    }
    else if (shape == JsonSchemaChildShape::Array && value.is_array())
    {
        const auto& array = value.as_array();
        for (std::size_t index = 0; index < array.size(); ++index)
        {
            visitor(array[index], ChildPath(path, std::to_string(index)), true);
        }
    }
    else if ((shape == JsonSchemaChildShape::Map || shape == JsonSchemaChildShape::Dependencies) &&
             value.is_object())
    {
        for (const auto& member : value.as_object())
        {
            if (shape != JsonSchemaChildShape::Dependencies || !member.value().is_array())
            {
                visitor(member.value(), ChildPath(path, member.key()), false);
            }
        }
    }
}

} // namespace rad::detail
