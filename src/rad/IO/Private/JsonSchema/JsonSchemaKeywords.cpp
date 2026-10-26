#include "JsonSchemaKeywords.h"

#include <algorithm>
#include <iterator>

namespace rad::detail
{
namespace
{

using Vocabulary = JsonSchemaKeywordVocabulary;
using Shape = JsonSchemaChildShape;

// Keep the reference index's traversal order stable for identifier diagnostics.
constexpr JsonSchemaKeyword ChildKeywords[] = {
    {"definitions", Vocabulary::Core, {Shape::Map, Shape::Unsupported, Shape::Unsupported}, true},
    {"$defs", Vocabulary::Core, {Shape::Unsupported, Shape::Map, Shape::Map}},
    {"properties", Vocabulary::Applicator, {Shape::Map, Shape::Map, Shape::Map}},
    {"patternProperties", Vocabulary::Applicator, {Shape::Map, Shape::Map, Shape::Map}},
    {"dependencies",
     Vocabulary::Applicator,
     {Shape::Dependencies, Shape::Unsupported, Shape::Unsupported}},
    {"dependentSchemas", Vocabulary::Applicator, {Shape::Unsupported, Shape::Map, Shape::Map}},
    {"additionalProperties", Vocabulary::Applicator, {Shape::Schema, Shape::Schema, Shape::Schema}},
    {"propertyNames", Vocabulary::Applicator, {Shape::Schema, Shape::Schema, Shape::Schema}},
    {"contains", Vocabulary::Applicator, {Shape::Schema, Shape::Schema, Shape::Schema}},
    {"not", Vocabulary::Applicator, {Shape::Schema, Shape::Schema, Shape::Schema}},
    {"if", Vocabulary::Applicator, {Shape::Schema, Shape::Schema, Shape::Schema}},
    {"then", Vocabulary::Applicator, {Shape::Schema, Shape::Schema, Shape::Schema}},
    {"else", Vocabulary::Applicator, {Shape::Schema, Shape::Schema, Shape::Schema}},
    {"unevaluatedItems",
     Vocabulary::Unevaluated,
     {Shape::Unsupported, Shape::Schema, Shape::Schema}},
    {"unevaluatedProperties",
     Vocabulary::Unevaluated,
     {Shape::Unsupported, Shape::Schema, Shape::Schema}},
    {"additionalItems", Vocabulary::Applicator, {Shape::Schema, Shape::Schema, Shape::Unsupported}},
    {"allOf", Vocabulary::Applicator, {Shape::Array, Shape::Array, Shape::Array}},
    {"anyOf", Vocabulary::Applicator, {Shape::Array, Shape::Array, Shape::Array}},
    {"oneOf", Vocabulary::Applicator, {Shape::Array, Shape::Array, Shape::Array}},
    {"prefixItems", Vocabulary::Applicator, {Shape::Unsupported, Shape::Unsupported, Shape::Array}},
    {"items", Vocabulary::Applicator, {Shape::SchemaOrArray, Shape::SchemaOrArray, Shape::Schema}},
};

constexpr JsonSchemaKeyword LeafKeywords[] = {
    {"$id", Vocabulary::Core, {Shape::None, Shape::None, Shape::None}},
    {"$schema", Vocabulary::Core, {Shape::None, Shape::None, Shape::None}},
    {"$ref", Vocabulary::Core, {Shape::None, Shape::None, Shape::None}},
    {"$anchor", Vocabulary::Core, {Shape::Unsupported, Shape::None, Shape::None}},
    {"$vocabulary", Vocabulary::Core, {Shape::Unsupported, Shape::None, Shape::None}},
    {"$recursiveRef", Vocabulary::Core, {Shape::Unsupported, Shape::None, Shape::Unsupported}},
    {"$recursiveAnchor", Vocabulary::Core, {Shape::Unsupported, Shape::None, Shape::Unsupported}},
    {"$dynamicRef", Vocabulary::Core, {Shape::Unsupported, Shape::Unsupported, Shape::None}},
    {"$dynamicAnchor", Vocabulary::Core, {Shape::Unsupported, Shape::Unsupported, Shape::None}},
    {"type", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"enum", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"const", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"multipleOf", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"minimum", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"maximum", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"exclusiveMinimum", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"exclusiveMaximum", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"minLength", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"maxLength", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"pattern", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"minProperties", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"maxProperties", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"required", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"minItems", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"maxItems", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"uniqueItems", Vocabulary::Validation, {Shape::None, Shape::None, Shape::None}},
    {"dependentRequired", Vocabulary::Validation, {Shape::Unsupported, Shape::None, Shape::None}},
    {"minContains", Vocabulary::Validation, {Shape::Unsupported, Shape::None, Shape::None}},
    {"maxContains", Vocabulary::Validation, {Shape::Unsupported, Shape::None, Shape::None}},
};

constexpr auto Lookup = []
{
    std::array<const JsonSchemaKeyword*, std::size(ChildKeywords) + std::size(LeafKeywords)>
        result{};
    std::size_t index = 0;
    for (const auto& keyword : ChildKeywords)
    {
        result[index++] = &keyword;
    }
    for (const auto& keyword : LeafKeywords)
    {
        result[index++] = &keyword;
    }
    std::sort(result.begin(), result.end(),
              [](const auto* left, const auto* right) { return left->name < right->name; });
    return result;
}();

} // namespace

JsonSchemaChildShape JsonSchemaKeyword::Shape(JsonSchemaDialect dialect) const
{
    switch (dialect)
    {
    case JsonSchemaDialect::Auto:
        return JsonSchemaChildShape::Unsupported;
    case JsonSchemaDialect::Draft7:
        return shapes[0];
    case JsonSchemaDialect::Draft2019_09:
        return shapes[1];
    case JsonSchemaDialect::Draft2020_12:
        return shapes[2];
    }
    return JsonSchemaChildShape::Unsupported;
}

JsonSchemaChildShape JsonSchemaKeyword::CandidateShape() const
{
    auto result = JsonSchemaChildShape::Unsupported;
    for (const auto shape : shapes)
    {
        if (shape == JsonSchemaChildShape::SchemaOrArray)
        {
            return shape;
        }
        if (shape != JsonSchemaChildShape::Unsupported)
        {
            result = shape;
        }
    }
    return result;
}

bool JsonSchemaKeyword::AcceptsChild(JsonSchemaDialect dialect, bool arrayChild) const
{
    const auto shape = Shape(dialect);
    if (arrayChild)
    {
        return shape == JsonSchemaChildShape::Array || shape == JsonSchemaChildShape::SchemaOrArray;
    }
    return shape == JsonSchemaChildShape::Schema || shape == JsonSchemaChildShape::SchemaOrArray ||
           shape == JsonSchemaChildShape::Map || shape == JsonSchemaChildShape::Dependencies;
}

const JsonSchemaKeyword* FindJsonSchemaKeyword(std::string_view name)
{
    const auto found = std::lower_bound(Lookup.begin(), Lookup.end(), name,
                                        [](const auto* keyword, std::string_view value)
                                        { return keyword->name < value; });
    return found != Lookup.end() && (*found)->name == name ? *found : nullptr;
}

std::span<const JsonSchemaKeyword> JsonSchemaChildKeywords()
{
    return ChildKeywords;
}

} // namespace rad::detail
