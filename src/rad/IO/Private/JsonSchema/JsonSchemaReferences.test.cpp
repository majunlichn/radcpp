#include "JsonSchemaReferences.h"

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <utility>

TEST(IO, JsonSchemaIndexStringMap)
{
    const std::string keys[] = {
        "", "short", std::string(128, 'x'), "/properties/a~1b~0c", "\xc3\xa9\xe7\x8c\xab",
        std::string("key\0tail", 8),
    };
    rad::detail::JsonSchemaStringMap<int> index;
    int number = 0;
    for (const auto& key : keys)
    {
        index.emplace(key, number++);
    }
    index.emplace("key", -1);
    for (const auto& key : keys)
    {
        const auto owned = index.find(key);
        ASSERT_NE(owned, index.end());
        const auto padded = key + "/not-part-of-the-view";
        const auto view = std::string_view(padded).substr(0, key.size());
        const auto borrowed = index.find(view);
        ASSERT_NE(borrowed, index.end());
        EXPECT_EQ(borrowed, owned);
        EXPECT_EQ(index.hash_function()(key), index.hash_function()(view));
        EXPECT_TRUE(index.contains(view));
        EXPECT_EQ(index.find(std::string_view(padded)), index.end());
    }
    EXPECT_EQ(index.find("short")->second, index.find(std::string_view("short"))->second);
    EXPECT_EQ(index.find("key")->second, -1);
    EXPECT_EQ(index.find(std::string_view{}), index.find(""));
}

TEST(IO, JsonSchemaIndexLookup)
{
    const std::string name = std::string(80, 'n') + "/~\xc3\xa9" + std::string("\0tail", 5);
    const auto prefix = name.substr(0, name.find('\0'));
    const std::string anchor(80, 'a');
    const std::string uri = "https://example.com/index.json";
    const rad::JsonValue schema = rad::JsonObject{
        {"$defs", rad::JsonObject{
            {prefix, rad::JsonObject{{"type", "null"}}},
            {name, rad::JsonObject{{"$id", "nested"}, {"$dynamicAnchor", anchor}}}}},
        {"properties", rad::JsonObject{{name, rad::JsonObject{{"$ref", "nested#" + anchor}}}}},
    };
    const rad::JsonSchemaCompileOptions options{
        .dialect = rad::JsonSchemaDialect::Draft2020_12, .retrievalUri = uri};
    auto compiled = rad::detail::JsonSchemaReferences::Compile(
        schema, options.dialect, options.maxCompilationDepth, options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    auto& references = compiled.value();
    const auto targetPath = rad::detail::ChildPath("/0/$defs", name);
    const auto prefixPath = rad::detail::ChildPath("/0/$defs", prefix);
    const auto sourcePath = rad::detail::ChildPath("/0/properties", name);
    const auto referencePath = rad::detail::ChildPath(sourcePath, "$ref");
    const auto patternPath = rad::detail::ChildPath(targetPath, "pattern");
    auto pattern = rad::detail::JsonSchemaRegex::Compile("^x$");
    ASSERT_TRUE(pattern);
    rad::detail::JsonSchemaReferences::Patterns patterns;
    patterns.emplace(patternPath, std::move(pattern.value()));
    references.SetPatterns(std::move(patterns));

    const auto check = [&](std::string_view path, bool found)
    {
        const auto* profile = references.Profile(path);
        const auto* resource = references.Resource(path);
        ASSERT_EQ(profile != nullptr, found);
        ASSERT_EQ(resource != nullptr, found);
        if (found)
        {
            EXPECT_EQ(profile->dialect, options.dialect);
            EXPECT_EQ(*resource, targetPath);
            const auto* anchors = references.DynamicAnchors(path);
            ASSERT_NE(anchors, nullptr);
            ASSERT_EQ(anchors->count(anchor), 1);
            EXPECT_EQ(anchors->at(anchor), targetPath);
        }
        else
        {
            EXPECT_EQ(references.DynamicAnchors(path), nullptr);
        }
    };
    check(targetPath, true);
    const auto paddedTarget = targetPath + "/not-part-of-the-view";
    check(std::string_view(paddedTarget).substr(0, targetPath.size()), true);
    check(paddedTarget, false);
    check({}, false);
    const auto* prefixResource = references.Resource(prefixPath);
    ASSERT_NE(prefixResource, nullptr);
    EXPECT_EQ(*prefixResource, "/0");
    EXPECT_EQ(references.DynamicAnchors(prefixPath), nullptr);

    const auto paddedReference = referencePath + "/missing";
    const auto* resolution =
        references.Find(std::string_view(paddedReference).substr(0, referencePath.size()));
    ASSERT_NE(resolution, nullptr);
    ASSERT_TRUE(*resolution);
    EXPECT_EQ(resolution->value().path, targetPath);
    EXPECT_EQ(resolution->value().dynamicAnchor, "");
    ASSERT_NE(resolution->value().schema, nullptr);
    EXPECT_EQ(resolution->value().schema,
              rad::detail::FindJsonSchemaValue(references.Documents(), targetPath));
    EXPECT_EQ(references.Find(paddedReference), nullptr);
    EXPECT_EQ(references.Find({}), nullptr);

    const auto paddedPattern = patternPath + "/missing";
    const auto* expression =
        references.FindPattern(std::string_view(paddedPattern).substr(0, patternPath.size()));
    ASSERT_NE(expression, nullptr);
    const auto matches = expression->Matches("x", {});
    ASSERT_TRUE(matches);
    EXPECT_TRUE(matches.value());
    EXPECT_EQ(references.FindPattern(paddedPattern), nullptr);
    EXPECT_EQ(references.FindPattern({}), nullptr);
}

TEST(IO, JsonSchemaBoundTargetsSurviveMoves)
{
    using References = rad::detail::JsonSchemaReferences;
    rad::JsonSchemaCompileOptions options{.dialect = rad::JsonSchemaDialect::Draft2020_12};
    options.retrievalUri = "https://example.com/root.json";
    rad::JsonArray branches{
        rad::JsonObject{{"$ref", "#"}},
        rad::JsonObject{{"$ref", "#/$defs/yes"}},
        rad::JsonObject{{"$ref", "#/$defs/no"}},
    };
    for (std::size_t index = 0; index < 32; ++index)
    {
        const auto uri = "https://example.com/doc" + std::to_string(index) + ".json";
        options.documents.push_back({uri, rad::JsonObject{{"const", index}}});
        branches.push_back(rad::JsonObject{{"$ref", uri}});
    }
    rad::JsonValue schema = rad::JsonObject{
        {"$defs", rad::JsonObject{{"yes", true}, {"no", false}}},
        {"allOf", std::move(branches)},
    };
    auto compiled = References::Compile(
        schema, options.dialect, options.maxCompilationDepth, options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    schema = nullptr;
    options.documents.clear();

    const auto check = [](const References& references)
    {
        ASSERT_EQ(references.Documents().as_array().size(), 33U);
        for (std::size_t index = 0; index < 35; ++index)
        {
            const auto path = "/0/allOf/" + std::to_string(index) + "/$ref";
            const auto* resolution = references.Find(path);
            ASSERT_NE(resolution, nullptr);
            ASSERT_TRUE(*resolution);
            const auto& target = resolution->value();
            ASSERT_NE(target.schema, nullptr);
            EXPECT_EQ(target.schema,
                      rad::detail::FindJsonSchemaValue(references.Documents(), target.path));
            if (index == 0)
            {
                EXPECT_EQ(target.schema, &references.Documents().as_array()[0]);
            }
            else if (index < 3)
            {
                ASSERT_TRUE(target.schema->is_bool());
                EXPECT_EQ(target.schema->as_bool(), index == 1);
            }
            else
            {
                EXPECT_EQ(target.schema->as_object().at("const"), rad::JsonValue(index - 3));
            }
        }
    };
    check(compiled.value());
    const auto* root = compiled.value().Find("/0/allOf/0/$ref")->value().schema;
    auto moved = std::move(compiled.value());
    check(moved);
    EXPECT_EQ(moved.Find("/0/allOf/0/$ref")->value().schema, root);

    auto replacement = References::Compile(
        rad::JsonObject{{"$ref", "#"}}, options.dialect, options.maxCompilationDepth, options);
    ASSERT_TRUE(replacement);
    replacement.value() = std::move(moved);
    check(replacement.value());
    EXPECT_EQ(replacement.value().Find("/0/allOf/0/$ref")->value().schema, root);
}

TEST(IO, JsonSchemaIndexProfileErrors)
{
    const std::string name = std::string(80, 'n') + "/~\xc3\xa9" + std::string("\0tail", 5);
    const auto path = rad::detail::ChildPath("/0/$defs", name);
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const rad::JsonSchemaCompileOptions options{
            .dialect = dialect, .retrievalUri = "https://example.com/profile.json"};
        const auto compiled = rad::detail::JsonSchemaReferences::Compile(
            rad::JsonObject{{"$defs", rad::JsonObject{
                {name, rad::JsonObject{{"$schema", 42}}}}}},
            dialect, options.maxCompilationDepth, options);
        ASSERT_TRUE(compiled) << compiled.error().message;
        const auto& references = compiled.value();
        const auto* error = references.ProfileError(path);
        ASSERT_NE(error, nullptr);
        EXPECT_EQ(error->code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
        EXPECT_EQ(error->schemaUri, options.retrievalUri);
        EXPECT_EQ(error->schemaPath, rad::detail::ChildPath(path.substr(2), "$schema"));

        const auto descendant = path + "/properties/child";
        const auto padded = descendant + "/not-part-of-the-view";
        EXPECT_EQ(references.ProfileError(descendant), error);
        EXPECT_EQ(references.ProfileError(
                      std::string_view(padded).substr(0, descendant.size())), error);
        EXPECT_EQ(references.ProfileError(path + "-other/properties/child"), nullptr);
        EXPECT_EQ(references.ProfileError(path.substr(0, path.find('\0'))), nullptr);
        EXPECT_EQ(references.ProfileError("/0"), nullptr);
        EXPECT_EQ(references.ProfileError({}), nullptr);
    }
}
