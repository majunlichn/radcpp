#include <rad/IO/JsonSchema.h>
#include <rad/IO/JsonSchema.test.h>

#include <gtest/gtest.h>

#include <format>
#include <string>
#include <string_view>

using rad::test::MakeJsonDocument;

TEST(JsonSchemaDialects, DeclaredSelection)
{
    struct Draft
    {
        rad::JsonSchemaDialect dialect;
        std::string_view uri;
    };
    constexpr Draft drafts[] = {
        {rad::JsonSchemaDialect::Draft7, "http://json-schema.org/draft-07/schema#"},
        {rad::JsonSchemaDialect::Draft2019_09, "https://json-schema.org/draft/2019-09/schema"},
        {rad::JsonSchemaDialect::Draft2020_12, "https://json-schema.org/draft/2020-12/schema"},
    };
    for (const auto& draft : drafts)
    {
        const rad::JsonValue schema = rad::JsonObject{{"$schema", draft.uri}, {"type", "integer"}};
        for (const auto selected :
             {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
              rad::JsonSchemaDialect::Draft2020_12})
        {
            const auto compiled =
                rad::JsonSchema::Compile(MakeJsonDocument(schema), {.dialect = selected});
            ASSERT_TRUE(compiled) << compiled.error().message;
            EXPECT_EQ(compiled.value().Dialect(), draft.dialect);
            EXPECT_TRUE(compiled.value().Validate(1));
            EXPECT_FALSE(compiled.value().Validate("invalid"));
        }
        const auto inferred = rad::JsonSchema::Compile(MakeJsonDocument(schema));
        ASSERT_TRUE(inferred);
        EXPECT_EQ(inferred.value().Dialect(), draft.dialect);
    }
}

TEST(JsonSchemaDialects, DefaultSelection)
{
    for (const rad::JsonValue schema :
         {rad::JsonValue(true), rad::JsonValue(false), rad::JsonValue(rad::JsonObject{})})
    {
        const auto compiled = rad::JsonSchema::Compile(MakeJsonDocument(schema));
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_EQ(compiled.value().Dialect(), rad::JsonSchemaDialect::Draft2020_12);
        EXPECT_EQ(static_cast<bool>(compiled.value().Validate(1)),
                  !schema.is_bool() || schema.as_bool());
    }
    const auto schema = rad::JsonObject{{"prefixItems", rad::JsonArray{false}}};
    const auto compiled = rad::JsonSchema::Compile(MakeJsonDocument(schema));
    ASSERT_TRUE(compiled) << compiled.error().message;
    EXPECT_TRUE(compiled.value().Validate(rad::JsonArray{}));
    EXPECT_FALSE(compiled.value().Validate(rad::JsonArray{1}));
    for (const auto selected :
         {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
          rad::JsonSchemaDialect::Draft2020_12})
    {
        const auto explicitDefault =
            rad::JsonSchema::Compile(MakeJsonDocument(schema), {.dialect = selected});
        ASSERT_TRUE(explicitDefault) << explicitDefault.error().message;
        EXPECT_EQ(explicitDefault.value().Dialect(), selected);
        EXPECT_EQ(static_cast<bool>(explicitDefault.value().Validate(rad::JsonArray{1})),
                  selected != rad::JsonSchemaDialect::Draft2020_12);
    }
    const auto invalidDialect = rad::JsonSchema::Compile(
        MakeJsonDocument(true), {.dialect = static_cast<rad::JsonSchemaDialect>(999)});
    ASSERT_FALSE(invalidDialect);
    EXPECT_EQ(invalidDialect.error().code, rad::JsonSchemaCompileErrorCode::InvalidDialect);
}

TEST(JsonSchemaDialects, RootDeclarationPrecedesReferenceSemantics)
{
    struct Draft
    {
        rad::JsonSchemaDialect dialect;
        std::string_view uri;
        std::string_view definitions;
    };
    constexpr Draft drafts[] = {
        {rad::JsonSchemaDialect::Draft7, "http://json-schema.org/draft-07/schema", "definitions"},
        {rad::JsonSchemaDialect::Draft2019_09, "https://json-schema.org/draft/2019-09/schema",
         "$defs"},
        {rad::JsonSchemaDialect::Draft2020_12, "https://json-schema.org/draft/2020-12/schema",
         "$defs"},
    };
    const std::string rootUri = "https://example.com/root.json";
    for (const auto& declared : drafts)
    {
        SCOPED_TRACE(declared.uri);
        const auto root = rad::JsonObject{
            {"$schema", declared.uri},
            {"$ref", "#/" + std::string(declared.definitions) + "/value"},
            {declared.definitions,
             rad::JsonObject{{"value", rad::JsonObject{{"type", "integer"}}}}},
            {"type", "string"},
        };
        for (const auto selected :
             {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
              rad::JsonSchemaDialect::Draft2020_12})
        {
            SCOPED_TRACE(static_cast<int>(selected));
            const auto compiled =
                rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), {.dialect = selected});
            ASSERT_TRUE(compiled) << compiled.error().message;
            EXPECT_EQ(compiled.value().Dialect(), declared.dialect);
            EXPECT_EQ(static_cast<bool>(compiled.value().Validate(1)),
                      declared.dialect == rad::JsonSchemaDialect::Draft7);
            EXPECT_FALSE(compiled.value().Validate("invalid"));
        }
    }
}

TEST(JsonSchemaDialects, InvalidRootDeclarationBesideReference)
{
    struct Case
    {
        rad::JsonValue declaration;
        rad::JsonSchemaCompileErrorCode code;
    };
    const Case cases[] = {
        {42, rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {"relative-meta", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {"https://example.com/unregistered-meta",
         rad::JsonSchemaCompileErrorCode::UnsupportedFeature},
    };
    const std::string rootUri = "https://example.com/root.json";
    for (const auto& testCase : cases)
    {
        const auto root = rad::JsonObject{{"$schema", testCase.declaration}, {"$ref", "#"}};
        for (const auto selected :
             {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
              rad::JsonSchemaDialect::Draft2020_12})
        {
            const auto compiled =
                rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), {.dialect = selected});
            ASSERT_FALSE(compiled);
            EXPECT_EQ(compiled.error().code, testCase.code);
            EXPECT_EQ(compiled.error().schemaUri, rootUri);
            EXPECT_EQ(compiled.error().schemaPath, "/$schema");
        }
    }
}

TEST(JsonSchemaDialects, DeclarationsRequireResourceRoots)
{
    for (const auto uri : {"https://json-schema.org/draft/2019-09/schema",
                           "https://json-schema.org/draft/2020-12/schema"})
    {
        SCOPED_TRACE(uri);
        const std::string rootUri = "https://example.com/root.json";
        const auto child = rad::JsonObject{{"$schema", uri}, {"type", "integer"}};
        auto root =
            rad::JsonObject{{"$schema", uri}, {"properties", rad::JsonObject{{"value", child}}}};
        const auto invalid = rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri));
        ASSERT_FALSE(invalid);
        EXPECT_EQ(invalid.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
        EXPECT_EQ(invalid.error().schemaUri, rootUri);
        EXPECT_EQ(invalid.error().schemaPath, "/properties/value/$schema");

        root.at("properties").as_object().at("value").as_object()["$id"] = "value";
        const auto compiled = rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri));
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(rad::JsonObject{{"value", 1}}));
        EXPECT_FALSE(compiled.value().Validate(rad::JsonObject{{"value", "invalid"}}));

        rad::JsonSchemaCompileOptions options;
        options.documents = {
            MakeJsonDocument(
                rad::JsonObject{
                    {"$schema", uri},
                    {"properties",
                     rad::JsonObject{
                         {"value",
                          rad::JsonObject{
                              {"$schema", uri},
                              {"$defs",
                               rad::JsonObject{
                                   {"meta", rad::JsonObject{{"$id", "https://example.com/meta"},
                                                            {"$schema", uri}}}}}}}}}},
                "https://example.com/wrapper.json"),
        };
        // Resolving the embedded meta-schema also checks its containing schema's dialect.
        const auto invalidMeta = rad::JsonSchema::Compile(
            MakeJsonDocument(rad::JsonObject{{"$schema", "https://example.com/meta"}}, rootUri),
            options);
        ASSERT_FALSE(invalidMeta);
        EXPECT_EQ(invalidMeta.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
        EXPECT_EQ(invalidMeta.error().schemaUri, options.documents[0]->Uri());
        EXPECT_EQ(invalidMeta.error().schemaPath, "/properties/value/$schema");
    }
}

TEST(JsonSchemaDialects, StandardMetaSchemaDiagnostics)
{
    struct Draft
    {
        rad::JsonSchemaDialect dialect;
        std::string_view uri;
        std::string_view definition;
    };
    constexpr Draft drafts[] = {
        {rad::JsonSchemaDialect::Draft7, "http://json-schema.org/draft-07/schema",
         "/definitions/nonNegativeInteger"},
        {rad::JsonSchemaDialect::Draft2019_09,
         "https://json-schema.org/draft/2019-09/meta/validation", "/$defs/nonNegativeInteger"},
        {rad::JsonSchemaDialect::Draft2020_12,
         "https://json-schema.org/draft/2020-12/meta/validation", "/$defs/nonNegativeInteger"},
    };
    for (const auto& draft : drafts)
    {
        SCOPED_TRACE(draft.uri);
        const auto compiled = rad::JsonSchema::Compile(
            MakeJsonDocument(rad::JsonObject{
                {"$ref", std::string(draft.uri) + "#" + std::string(draft.definition)}}),
            {.dialect = draft.dialect});
        ASSERT_TRUE(compiled) << compiled.error().schemaUri << compiled.error().schemaPath << ": "
                              << compiled.error().message;
        const auto invalid = compiled.value().Validate(-1);
        ASSERT_FALSE(invalid);
        ASSERT_EQ(invalid.errors.size(), 1);
        EXPECT_EQ(invalid.errors[0].instancePath, "");
        EXPECT_EQ(invalid.errors[0].schemaPath,
                  std::string(draft.definition) + "/minimum");
        EXPECT_EQ(invalid.errors[0].schemaUri, draft.uri);
    }
}

TEST(JsonSchemaDialects, StandardMetaSchemaPrecedence)
{
    struct Draft
    {
        rad::JsonSchemaDialect dialect;
        std::string_view uri;
    };
    constexpr Draft drafts[] = {
        {rad::JsonSchemaDialect::Draft7, "http://json-schema.org/draft-07/schema"},
        {rad::JsonSchemaDialect::Draft2019_09, "https://json-schema.org/draft/2019-09/schema"},
        {rad::JsonSchemaDialect::Draft2020_12, "https://json-schema.org/draft/2020-12/schema"},
    };
    for (const auto& draft : drafts)
    {
        SCOPED_TRACE(draft.uri);
        const auto root = rad::JsonObject{{"$ref", draft.uri}};
        for (const bool alias : {false, true})
        {
            rad::JsonSchemaCompileOptions options{.dialect = draft.dialect};
            options.documents = {
                MakeJsonDocument(rad::JsonObject{{"$id", draft.uri}, {"const", "caller"}},
                                 alias ? "https://example.com/meta.json" : std::string(draft.uri)),
            };
            const auto compiled = rad::JsonSchema::Compile(MakeJsonDocument(root), options);
            ASSERT_TRUE(compiled) << compiled.error().message;
            EXPECT_TRUE(compiled.value().Validate("caller"));
            const auto invalid = compiled.value().Validate(rad::JsonObject{});
            ASSERT_FALSE(invalid);
            ASSERT_EQ(invalid.errors.size(), 1);
            EXPECT_EQ(invalid.errors[0].schemaPath, "/const");
            EXPECT_EQ(invalid.errors[0].schemaUri, options.documents[0]->Uri());
        }
        const auto local = rad::ParseJson(
            std::format(R"json({{"$id": "{}", "type": "object",
                               "properties": {{"value": {{"$ref": "{}"}}}}}})json",
                        draft.uri, draft.uri));
        ASSERT_TRUE(local);
        const auto compiled =
            rad::JsonSchema::Compile(MakeJsonDocument(local.value()), {.dialect = draft.dialect});
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(
            rad::JsonObject{{"value", rad::JsonObject{{"type", 7}}}}));

        if (draft.dialect != rad::JsonSchemaDialect::Draft7)
        {
            rad::JsonSchemaCompileOptions options{.dialect = draft.dialect};
            const auto uri = std::string(draft.uri.substr(0, draft.uri.rfind('/'))) +
                             "/meta/validation";
            options.documents = {
                MakeJsonDocument(rad::JsonObject{{"const", rad::JsonObject{{"type", 42}}}}, uri),
            };
            const auto incomplete = rad::JsonSchema::Compile(MakeJsonDocument(root), options);
            ASSERT_FALSE(incomplete);
            EXPECT_EQ(incomplete.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
            EXPECT_EQ(incomplete.error().schemaUri, draft.uri);
            auto updatedSchema = options.documents[0]->Root();
            updatedSchema.as_object()["$defs"] = rad::JsonObject{{"stringArray", true}};
            options.documents[0] = MakeJsonDocument(std::move(updatedSchema), uri);
            const auto overridden = rad::JsonSchema::Compile(MakeJsonDocument(root), options);
            ASSERT_TRUE(overridden) << overridden.error().message;
            EXPECT_TRUE(overridden.value().Validate(rad::JsonObject{{"type", 42}}));
        }
    }
    const auto mismatched = rad::JsonSchema::Compile(
        MakeJsonDocument(rad::JsonObject{{"$ref", "http://json-schema.org/draft-07/schema"}}),
        {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_FALSE(mismatched);
    EXPECT_EQ(mismatched.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    EXPECT_EQ(mismatched.error().schemaPath, "/$schema");
    EXPECT_EQ(mismatched.error().schemaUri, "http://json-schema.org/draft-07/schema");
}

TEST(JsonSchemaDialects, VocabularyDeclarations)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const std::string prefix = dialect == rad::JsonSchemaDialect::Draft2019_09
                                       ? "https://json-schema.org/draft/2019-09/vocab/"
                                       : "https://json-schema.org/draft/2020-12/vocab/";
        const auto core = prefix + "core";
        const std::string rootUri = "https://example.com/root.json";
        rad::JsonSchemaCompileOptions options{.dialect = dialect};
        const auto base = rad::JsonObject{
            {"$vocabulary", rad::JsonObject{{core, true}, {"urn:example:optional", false}}},
            {"type", "integer"}, {"minimum", 2},
        };
        const auto compiled = rad::JsonSchema::Compile(MakeJsonDocument(base, rootUri), options);
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(2));
        EXPECT_FALSE(compiled.value().Validate(1));
        EXPECT_FALSE(compiled.value().Validate("invalid"));

        struct Case
        {
            std::string_view uri;
            rad::JsonValue required;
            rad::JsonSchemaCompileErrorCode code;
        };
        const Case cases[] = {
            {"urn:example:required", true, rad::JsonSchemaCompileErrorCode::UnsupportedFeature},
            {"relative", false, rad::JsonSchemaCompileErrorCode::InvalidSchema},
            {"HTTPS://example.com/optional", false, rad::JsonSchemaCompileErrorCode::InvalidSchema},
            {"urn:example:optional", 1, rad::JsonSchemaCompileErrorCode::InvalidSchema},
        };
        for (const auto& testCase : cases)
        {
            SCOPED_TRACE(testCase.uri);
            auto schema = base;
            schema.at("$vocabulary").as_object()[testCase.uri] = testCase.required;
            const auto invalid =
                rad::JsonSchema::Compile(MakeJsonDocument(schema, rootUri), options);
            ASSERT_FALSE(invalid);
            EXPECT_EQ(invalid.error().code, testCase.code);
            EXPECT_EQ(invalid.error().schemaPath,
                      testCase.uri.starts_with("HTTPS:")
                          ? "/$vocabulary/HTTPS:~1~1example.com~1optional"
                          : "/$vocabulary/" + std::string(testCase.uri));
            EXPECT_EQ(invalid.error().schemaUri, rootUri);
        }
        auto format = base;
        const auto formatUri =
            prefix +
            (dialect == rad::JsonSchemaDialect::Draft2019_09 ? "format" : "format-assertion");
        format.at("$vocabulary").as_object()[formatUri] = false;
        EXPECT_TRUE(rad::JsonSchema::Compile(MakeJsonDocument(format), {.dialect = dialect}));
        format.at("$vocabulary").as_object()[formatUri] = true;
        const auto requiredFormat =
            rad::JsonSchema::Compile(MakeJsonDocument(format), {.dialect = dialect});
        ASSERT_FALSE(requiredFormat);
        EXPECT_EQ(requiredFormat.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);

        for (const rad::JsonValue vocabulary :
             {rad::JsonValue(1), rad::JsonValue(rad::JsonObject{}),
              rad::JsonValue(rad::JsonObject{{core, false}})})
        {
            const auto invalid = rad::JsonSchema::Compile(
                MakeJsonDocument(rad::JsonObject{{"$vocabulary", vocabulary}}, rootUri), options);
            ASSERT_FALSE(invalid);
            EXPECT_EQ(invalid.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
            EXPECT_EQ(invalid.error().schemaPath,
                      vocabulary.is_object() && vocabulary.as_object().contains(core)
                          ? "/$vocabulary/https:~1~1json-schema.org~1draft~1" +
                                std::string(dialect == rad::JsonSchemaDialect::Draft2019_09
                                                ? "2019-09"
                                                : "2020-12") +
                                "~1vocab~1core"
                          : "/$vocabulary");
            EXPECT_EQ(invalid.error().schemaUri, rootUri);
        }
    }
    const auto ignored =
        rad::JsonSchema::Compile(MakeJsonDocument(rad::JsonObject{{"$vocabulary", 42}}),
                                 {.dialect = rad::JsonSchemaDialect::Draft7});
    ASSERT_TRUE(ignored) << ignored.error().message;
    EXPECT_TRUE(ignored.value().Validate(1));
}

TEST(JsonSchemaDialects, MetaSchemaCompilationDepth)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const std::string standard = dialect == rad::JsonSchemaDialect::Draft2019_09
                                         ? "https://json-schema.org/draft/2019-09/schema"
                                         : "https://json-schema.org/draft/2020-12/schema";
        const std::string rootUri = "https://example.com/root.json";
        rad::JsonSchemaCompileOptions options;
        options.documents = {
            MakeJsonDocument(rad::JsonObject{{"$schema", "https://example.com/meta-b.json"}},
                             "https://example.com/meta-a.json"),
            MakeJsonDocument(rad::JsonObject{{"$schema", standard}},
                             "https://example.com/meta-b.json"),
        };
        const rad::JsonValue schema =
            rad::JsonObject{{"$schema", "https://example.com/meta-a.json"}, {"type", "integer"}};
        struct Limit
        {
            std::size_t depth;
            std::string_view uri;
            std::string_view path;
        };
        constexpr Limit limits[] = {
            {0, "https://example.com/meta-a.json", ""},
            {1, "https://example.com/meta-a.json", "/$schema"},
            {2, "https://example.com/meta-b.json", ""},
        };
        for (const auto& limit : limits)
        {
            SCOPED_TRACE(limit.depth);
            options.maxCompilationDepth = limit.depth;
            const auto limited =
                rad::JsonSchema::Compile(MakeJsonDocument(schema, rootUri), options);
            ASSERT_FALSE(limited);
            EXPECT_EQ(limited.error().code, rad::JsonSchemaCompileErrorCode::DepthLimitExceeded);
            EXPECT_EQ(limited.error().schemaUri, limit.uri);
            EXPECT_EQ(limited.error().schemaPath, limit.path);
        }
        options.maxCompilationDepth = 3;
        const auto compiled = rad::JsonSchema::Compile(MakeJsonDocument(schema, rootUri), options);
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_EQ(compiled.value().Dialect(), dialect);
        EXPECT_TRUE(compiled.value().Validate(1));
        EXPECT_FALSE(compiled.value().Validate("invalid"));

        options.documents = {
            MakeJsonDocument(
                rad::JsonObject{
                    {"$schema", standard},
                    {"$defs",
                     rad::JsonObject{
                         {"meta", rad::JsonObject{{"$id", "https://example.com/embedded"},
                                                  {"$schema", standard}}}}},
                },
                "https://example.com/container.json"),
        };
        const rad::JsonValue embedded =
            rad::JsonObject{{"$schema", "https://example.com/embedded"}, {"type", "integer"}};
        options.maxCompilationDepth = 0;
        const auto resourceLimited =
            rad::JsonSchema::Compile(MakeJsonDocument(embedded, rootUri), options);
        ASSERT_FALSE(resourceLimited);
        EXPECT_EQ(resourceLimited.error().code,
                  rad::JsonSchemaCompileErrorCode::DepthLimitExceeded);
        EXPECT_EQ(resourceLimited.error().schemaUri, "https://example.com/container.json");
        EXPECT_EQ(resourceLimited.error().schemaPath, "/$defs/meta");
        options.maxCompilationDepth = 1;
        EXPECT_TRUE(rad::JsonSchema::Compile(MakeJsonDocument(embedded, rootUri), options));
    }
}

TEST(JsonSchemaDialects, CustomSelection)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const std::string base = dialect == rad::JsonSchemaDialect::Draft2019_09
                                     ? "https://json-schema.org/draft/2019-09/"
                                     : "https://json-schema.org/draft/2020-12/";
        const std::string rootUri = "https://example.com/root.json";
        rad::JsonSchemaCompileOptions options;
        options.documents = {
            MakeJsonDocument(rad::JsonObject{{"$id", "https://example.com/applicator"},
                                             {"$schema", base + "schema"},
                                             {"$vocabulary",
                                              rad::JsonObject{{base + "vocab/core", true},
                                                              {base + "vocab/applicator", true}}}},
                             "https://example.com/meta-a.json"),
            MakeJsonDocument(rad::JsonObject{{"$id", "https://example.com/validation"},
                                             {"$schema", base + "schema"},
                                             {"$vocabulary",
                                              rad::JsonObject{{base + "vocab/core", true},
                                                              {base + "vocab/validation", false}}}},
                             "https://example.com/meta-b.json"),
        };
        const auto loose = rad::ParseJson(R"json({
            "$schema": "https://example.com/applicator",
            "type": 42, "enum": [], "required": false, "pattern": 1,
            "minLength": false, "maxLength": false,
            "minimum": false, "minContains": "ignored", "uniqueItems": 1,
            "dependentRequired": 4, "properties": {"value": {"minimum": false}}
        })json");
        ASSERT_TRUE(loose);
        const auto inferred =
            rad::JsonSchema::Compile(MakeJsonDocument(loose.value(), rootUri), options);
        ASSERT_TRUE(inferred) << inferred.error().schemaUri << inferred.error().schemaPath << ": "
                              << inferred.error().message;
        EXPECT_EQ(inferred.value().Dialect(), dialect);
        EXPECT_TRUE(inferred.value().Validate(rad::JsonObject{{"value", 1}}));
        EXPECT_TRUE(inferred.value().Validate(""));
        EXPECT_TRUE(inferred.value().Validate(rad::JsonValue(std::string(4096, 'a'))));
        for (const auto selected :
             {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
              rad::JsonSchemaDialect::Draft2020_12})
        {
            options.dialect = selected;
            const auto declared =
                rad::JsonSchema::Compile(MakeJsonDocument(loose.value(), rootUri), options);
            ASSERT_TRUE(declared) << declared.error().message;
            EXPECT_EQ(declared.value().Dialect(), dialect);
            EXPECT_TRUE(declared.value().Validate("ignored"));
        }
        options.dialect = rad::JsonSchemaDialect::Draft2020_12;

        const auto strict = rad::ParseJson(R"json({
            "$schema": "https://example.com/validation", "type": "integer", "minimum": 1,
            "allOf": 5, "items": 5, "unevaluatedProperties": 5,
            "properties": {"bad": {"$id": 5, "$ref": "unregistered.json"}}
        })json");
        ASSERT_TRUE(strict);
        const auto compiled =
            rad::JsonSchema::Compile(MakeJsonDocument(strict.value(), rootUri), options);
        ASSERT_TRUE(compiled) << compiled.error().schemaPath << ": " << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(1));
        EXPECT_FALSE(compiled.value().Validate(0));
        EXPECT_FALSE(compiled.value().Validate("invalid"));
        auto malformed = strict.value();
        malformed.as_object()["type"] = 42;
        const auto invalid =
            rad::JsonSchema::Compile(MakeJsonDocument(malformed, rootUri), options);
        ASSERT_FALSE(invalid);
        EXPECT_EQ(invalid.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
        EXPECT_EQ(invalid.error().schemaPath, "/type");
        EXPECT_EQ(invalid.error().schemaUri, rootUri);

        auto pointed = strict.value();
        pointed.as_object()["$ref"] = "#/properties/bad";
        const auto badTarget =
            rad::JsonSchema::Compile(MakeJsonDocument(pointed, rootUri), options);
        ASSERT_FALSE(badTarget);
        EXPECT_EQ(badTarget.error().schemaPath, "/properties/bad/$id");

        options.documents.push_back(
            MakeJsonDocument(rad::JsonObject{{"$schema", "https://example.com/validation"}},
                             "https://example.com/defaults"));
        const auto defaults = rad::JsonSchema::Compile(
            MakeJsonDocument(rad::JsonObject{{"$schema", "https://example.com/defaults"},
                                             {"type", "object"},
                                             {"properties", rad::JsonObject{{"value", false}}}},
                             rootUri),
            options);
        ASSERT_TRUE(defaults) << defaults.error().message;
        EXPECT_TRUE(defaults.value().Validate(rad::JsonObject{}));
        EXPECT_FALSE(defaults.value().Validate(rad::JsonObject{{"value", 1}}));
        EXPECT_FALSE(defaults.value().Validate(1));
    }
}

TEST(JsonSchemaDialects, CustomMetaSchemaDefaults)
{
    const auto root = rad::JsonObject{{"$schema", "https://example.com/meta"}, {"type", "integer"}};
    rad::JsonSchemaCompileOptions options;
    options.documents = {
        MakeJsonDocument(rad::JsonObject{}, "https://example.com/meta"),
    };
    const auto compiled = rad::JsonSchema::Compile(MakeJsonDocument(root), options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    EXPECT_EQ(compiled.value().Dialect(), rad::JsonSchemaDialect::Draft2020_12);
    EXPECT_TRUE(compiled.value().Validate(1));
    EXPECT_FALSE(compiled.value().Validate("invalid"));

    options.dialect = rad::JsonSchemaDialect::Draft2019_09;
    const auto explicitDefault = rad::JsonSchema::Compile(MakeJsonDocument(root), options);
    ASSERT_TRUE(explicitDefault) << explicitDefault.error().message;
    EXPECT_EQ(explicitDefault.value().Dialect(), rad::JsonSchemaDialect::Draft2019_09);
    EXPECT_TRUE(explicitDefault.value().Validate(1));
    EXPECT_FALSE(explicitDefault.value().Validate("invalid"));
}

TEST(JsonSchemaDialects, CustomCandidateShapes)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        const std::string base = dialect == rad::JsonSchemaDialect::Draft2019_09
                                     ? "https://json-schema.org/draft/2019-09/"
                                     : "https://json-schema.org/draft/2020-12/";
        const char* keyword =
            dialect == rad::JsonSchemaDialect::Draft2019_09 ? "items" : "prefixItems";
        const rad::JsonValue meta =
            rad::JsonObject{{"$id", "https://example.com/meta"},
                            {"$schema", base + "schema"},
                            {"$vocabulary", rad::JsonObject{{base + "vocab/core", true},
                                                            {base + "vocab/validation", false}}}};
        const std::string rootUri = "https://example.com/root.json";
        rad::JsonSchemaCompileOptions options;
        options.documents = {
            MakeJsonDocument(
                rad::JsonObject{{"$schema", base + "schema"}, {keyword, rad::JsonArray{meta}}},
                "https://example.com/container"),
        };
        const rad::JsonValue schema = rad::JsonObject{
            {"$schema", "https://example.com/meta"}, {"type", "integer"}, {"allOf", 5}};
        const auto compiled = rad::JsonSchema::Compile(MakeJsonDocument(schema, rootUri), options);
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(1));
        EXPECT_FALSE(compiled.value().Validate("invalid"));
        auto modifiedProfile = options.documents[0]->Root();
        modifiedProfile.as_object().at(keyword).as_array()[0].as_object()["$vocabulary"] =
            rad::JsonObject{};
        options.documents[0] =
            MakeJsonDocument(std::move(modifiedProfile), options.documents[0]->Uri());
        const auto missingCore =
            rad::JsonSchema::Compile(MakeJsonDocument(schema, rootUri), options);
        ASSERT_FALSE(missingCore);
        EXPECT_EQ(missingCore.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
        EXPECT_EQ(missingCore.error().schemaPath,
                  std::string("/") + keyword + "/0/$vocabulary/https:~1~1json-schema.org~1draft~1" +
                      (dialect == rad::JsonSchemaDialect::Draft2019_09 ? "2019-09" : "2020-12") +
                      "~1vocab~1core");
        EXPECT_EQ(missingCore.error().schemaUri, options.documents[0]->Uri());
        if (dialect == rad::JsonSchemaDialect::Draft2020_12)
        {
            auto updatedSchema = options.documents[0]->Root();
            updatedSchema.as_object().erase("prefixItems");
            updatedSchema.as_object()["items"] = rad::JsonArray{meta};
            options.documents[0] =
                MakeJsonDocument(std::move(updatedSchema), options.documents[0]->Uri());
            const auto hidden =
                rad::JsonSchema::Compile(MakeJsonDocument(schema, rootUri), options);
            ASSERT_FALSE(hidden);
            EXPECT_EQ(hidden.error().schemaPath, "/$schema");
            EXPECT_EQ(hidden.error().schemaUri, rootUri);
        }
    }
}

TEST(JsonSchemaDialects, CustomResourceProfiles)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        const std::string base = dialect == rad::JsonSchemaDialect::Draft2019_09
                                     ? "https://json-schema.org/draft/2019-09/"
                                     : "https://json-schema.org/draft/2020-12/";
        const std::string rootUri = "https://example.com/root.json";
        rad::JsonSchemaCompileOptions options;
        options.documents = {
            MakeJsonDocument(rad::JsonObject{{"$schema", base + "schema"},
                                             {"$vocabulary",
                                              rad::JsonObject{{base + "vocab/core", true},
                                                              {base + "vocab/applicator", true}}}},
                             "https://example.com/loose-meta"),
        };
        const auto schema = rad::ParseJson(std::format(R"json({{
            "$schema": "https://example.com/loose-meta",
            "$defs": {{
                "strict": {{"$id": "strict", "$schema": "{}schema", "type": "integer"}},
                "loose": {{"$id": "loose", "type": 42}}
            }},
            "properties": {{
                "strict": {{"$ref": "strict"}},
                "loose": {{"$ref": "loose"}}
            }}
        }})json", base));
        ASSERT_TRUE(schema);
        const auto compiled =
            rad::JsonSchema::Compile(MakeJsonDocument(schema.value(), rootUri), options);
        ASSERT_TRUE(compiled) << compiled.error().schemaPath << ": " << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(rad::JsonObject{{"strict", 1}, {"loose", "ignored"}}));
        const auto invalid = compiled.value().Validate(rad::JsonObject{{"strict", "invalid"}});
        ASSERT_FALSE(invalid);
        ASSERT_EQ(invalid.errors.size(), 1);
        EXPECT_EQ(invalid.errors[0].instancePath, "/strict");
        EXPECT_EQ(invalid.errors[0].schemaPath, "/$defs/strict/type");
        EXPECT_EQ(invalid.errors[0].schemaUri, rootUri);

        auto nonResource = schema.value();
        nonResource.as_object().at("properties").as_object().at("strict") =
            rad::JsonObject{{"$schema", base + "schema"}, {"type", "integer"}};
        const auto badDeclaration =
            rad::JsonSchema::Compile(MakeJsonDocument(nonResource, rootUri), options);
        ASSERT_FALSE(badDeclaration);
        EXPECT_EQ(badDeclaration.error().schemaPath, "/properties/strict/$schema");
        EXPECT_EQ(badDeclaration.error().schemaUri, rootUri);

        options.documents.push_back(
            MakeJsonDocument(rad::JsonObject{{"$schema", base + "schema"}, {"type", "integer"}},
                             "https://example.com/foreign.json"));
        auto referenced = schema.value();
        referenced.as_object()["$ref"] = "foreign.json";
        const auto external =
            rad::JsonSchema::Compile(MakeJsonDocument(referenced, rootUri), options);
        ASSERT_TRUE(external) << external.error().message;
        EXPECT_TRUE(external.value().Validate(1));
        EXPECT_FALSE(external.value().Validate("invalid"));
        auto modifiedForeign = options.documents.back()->Root();
        modifiedForeign.as_object()["$schema"] = "https://example.com/missing";
        options.documents.back() =
            MakeJsonDocument(std::move(modifiedForeign), options.documents.back()->Uri());
        const auto badExternal =
            rad::JsonSchema::Compile(MakeJsonDocument(referenced, rootUri), options);
        ASSERT_FALSE(badExternal);
        EXPECT_EQ(badExternal.error().schemaPath, "/$schema");
        EXPECT_EQ(badExternal.error().schemaUri, options.documents.back()->Uri());
    }
}

TEST(JsonSchemaDialects, CustomUnevaluatedVocabulary)
{
    rad::JsonSchemaCompileOptions options;
    options.documents = {
        MakeJsonDocument(
            rad::JsonObject{
                {"$schema", "https://json-schema.org/draft/2020-12/schema"},
                {"$vocabulary",
                 rad::JsonObject{
                     {"https://json-schema.org/draft/2020-12/vocab/core", true},
                     {"https://json-schema.org/draft/2020-12/vocab/unevaluated", false}}}},
            "https://example.com/meta"),
    };
    const auto compiled = rad::JsonSchema::Compile(
        MakeJsonDocument(rad::JsonObject{{"$schema", "https://example.com/meta"},
                                         {"properties", rad::JsonObject{{"value", true}}},
                                         {"prefixItems", rad::JsonArray{true}},
                                         {"unevaluatedProperties", false},
                                         {"unevaluatedItems", false}}),
        options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    EXPECT_TRUE(compiled.value().Validate(rad::JsonObject{}));
    EXPECT_TRUE(compiled.value().Validate(rad::JsonArray{}));
    EXPECT_FALSE(compiled.value().Validate(rad::JsonObject{{"value", 1}}));
    EXPECT_FALSE(compiled.value().Validate(rad::JsonArray{1}));
}

TEST(JsonSchemaDialects, UnevaluatedResourceProfileAnnotations)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const std::string base = dialect == rad::JsonSchemaDialect::Draft2019_09
                                     ? "https://json-schema.org/draft/2019-09/"
                                     : "https://json-schema.org/draft/2020-12/";
        const std::string strictUri = "https://example.com/strict.json";
        rad::JsonSchemaCompileOptions options{.dialect = dialect};
        options.documents = {
            MakeJsonDocument(
                rad::JsonObject{{"$schema", base + "schema"},
                                 {"$vocabulary",
                                  rad::JsonObject{{base + "vocab/core", true}}}},
                "https://example.com/loose-meta"),
            MakeJsonDocument(
                rad::JsonObject{
                    {"$schema", base + "schema"},
                    {"allOf", rad::JsonArray{
                        rad::JsonObject{{"properties", rad::JsonObject{{"known", true}}}}}},
                    {"unevaluatedProperties", false}},
                strictUri),
        };
        const auto compiled = rad::JsonSchema::Compile(
            MakeJsonDocument(
                rad::JsonObject{{"$schema", "https://example.com/loose-meta"},
                                 {"$ref", strictUri}, {"unevaluatedProperties", false}}),
            options);
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(rad::JsonObject{{"known", 1}}));
        const auto invalid =
            compiled.value().Validate(rad::JsonObject{{"known", 1}, {"extra", 2}});
        ASSERT_FALSE(invalid);
        ASSERT_EQ(invalid.errors.size(), 1);
        EXPECT_EQ(invalid.status, rad::JsonSchemaValidationStatus::Completed);
        EXPECT_EQ(invalid.errors[0].instancePath, "/extra");
        EXPECT_EQ(invalid.errors[0].schemaPath, "/unevaluatedProperties");
        EXPECT_EQ(invalid.errors[0].schemaUri, strictUri);
    }
}

TEST(JsonSchemaDialects, CustomDiagnostics)
{
    constexpr auto dialect = rad::JsonSchemaDialect::Draft2020_12;
    const std::string rootUri = "https://example.com/root.json";
    rad::JsonSchemaCompileOptions options;
    const auto root = rad::JsonObject{{"$schema", "https://example.com/meta"}};
    const auto missing = rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options);
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().schemaPath, "/$schema");
    EXPECT_EQ(missing.error().schemaUri, rootUri);
    options.documents = {
        MakeJsonDocument(rad::JsonObject{{"$schema", "https://example.com/other"}},
                         "https://example.com/meta"),
        MakeJsonDocument(rad::JsonObject{{"$schema", "https://example.com/meta"}},
                         "https://example.com/other"),
    };
    const auto cyclic = rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options);
    ASSERT_FALSE(cyclic);
    EXPECT_EQ(cyclic.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    EXPECT_EQ(cyclic.error().schemaPath, "/$schema");
    options.documents = {
        MakeJsonDocument(
            rad::JsonObject{
                {"$schema", "https://json-schema.org/draft/2020-12/schema"},
                {"$vocabulary",
                 rad::JsonObject{{"https://json-schema.org/draft/2020-12/vocab/core", true},
                                 {"urn:example:required", true}}},
            },
            "https://example.com/meta"),
    };
    const auto unsupported = rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options);
    ASSERT_FALSE(unsupported);
    EXPECT_EQ(unsupported.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    EXPECT_EQ(unsupported.error().schemaPath, "/$vocabulary/urn:example:required");
    EXPECT_EQ(unsupported.error().schemaUri, options.documents[0]->Uri());
    auto updatedSchema = options.documents[0]->Root();
    updatedSchema.as_object().at("$vocabulary").as_object()["urn:example:required"] = false;
    options.documents[0] = MakeJsonDocument(std::move(updatedSchema), options.documents[0]->Uri());
    EXPECT_TRUE(rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options));
    options.dialect = rad::JsonSchemaDialect::Draft2019_09;
    const auto declared = rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options);
    ASSERT_TRUE(declared) << declared.error().message;
    EXPECT_EQ(declared.value().Dialect(), dialect);
    options.dialect = dialect;
    EXPECT_TRUE(rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options));
}
