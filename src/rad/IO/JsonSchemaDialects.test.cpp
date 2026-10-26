#include <rad/IO/File.h>
#include <rad/IO/JsonSchema.h>

#include <boost/filesystem/operations.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <format>
#include <string>
#include <string_view>

TEST(IO, JsonSchemaAutomaticDialectSelection)
{
    EXPECT_EQ(rad::JsonSchemaCompileOptions{}.dialect, rad::JsonSchemaDialect::Auto);
    const auto path = std::filesystem::temp_directory_path() /
                      boost::filesystem::unique_path("rad-schema-%%%%-%%%%-%%%%.json").string();
    ASSERT_TRUE(rad::File::WriteText(path, "{}"));
    struct SchemaFile
    {
        rad::FilePath path;
        ~SchemaFile() { EXPECT_TRUE(rad::File::Remove(path)); }
    } file{path};
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
        for (const auto selected : {rad::JsonSchemaDialect::Auto, draft.dialect})
        {
            const auto compiled = rad::JsonSchema::Compile(schema, {.dialect = selected});
            ASSERT_TRUE(compiled) << compiled.error().message;
            EXPECT_EQ(compiled.value().Dialect(), draft.dialect);
        }
        const auto inferred = rad::JsonSchema::Compile(schema);
        ASSERT_TRUE(inferred);
        EXPECT_EQ(inferred.value().Dialect(), draft.dialect);
        ASSERT_TRUE(rad::File::WriteText(path, rad::PrettyJson(schema)));
        const auto fromFile = rad::JsonSchema::CompileFile(path);
        ASSERT_TRUE(fromFile) << fromFile.error().message;
        EXPECT_EQ(fromFile.value().Dialect(), draft.dialect);
        EXPECT_TRUE(fromFile.value().Validate(1));
        EXPECT_FALSE(fromFile.value().Validate("invalid"));
        const auto explicitFile = rad::JsonSchema::CompileFile(path, {.dialect = draft.dialect});
        ASSERT_TRUE(explicitFile) << explicitFile.error().message;
        EXPECT_EQ(explicitFile.value().Dialect(), draft.dialect);
    }
    for (const rad::JsonValue schema : {rad::JsonValue(true), rad::JsonValue(rad::JsonObject{})})
    {
        const auto automatic = rad::JsonSchema::Compile(schema);
        ASSERT_FALSE(automatic);
        EXPECT_EQ(automatic.error().code, rad::JsonSchemaCompileErrorCode::MissingDialect);
        EXPECT_TRUE(
            rad::JsonSchema::Compile(schema, {.dialect = rad::JsonSchemaDialect::Draft2020_12}));
    }
    ASSERT_TRUE(rad::File::WriteText(path, "true"));
    const auto missing = rad::JsonSchema::CompileFile(path);
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().code, rad::JsonSchemaCompileErrorCode::MissingDialect);
    ASSERT_TRUE(rad::File::WriteText(path, "{"));
    const auto invalidJson = rad::JsonSchema::CompileFile(path);
    ASSERT_FALSE(invalidJson);
    EXPECT_EQ(invalidJson.error().code, rad::JsonSchemaCompileErrorCode::InvalidJson);
    const auto invalidDialect =
        rad::JsonSchema::Compile(true, {.dialect = static_cast<rad::JsonSchemaDialect>(999)});
    ASSERT_FALSE(invalidDialect);
    EXPECT_EQ(invalidDialect.error().code, rad::JsonSchemaCompileErrorCode::InvalidDialect);
}

TEST(IO, JsonSchemaStandardMetaSchemaDiagnostics)
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
            rad::JsonObject{{"$ref", std::string(draft.uri) + "#" + std::string(draft.definition)}},
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

TEST(IO, JsonSchemaStandardMetaSchemaPrecedence)
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
                {alias ? "https://example.com/meta.json" : std::string(draft.uri),
                 rad::JsonObject{{"$id", draft.uri}, {"const", "caller"}}},
            };
            const auto compiled = rad::JsonSchema::Compile(root, options);
            ASSERT_TRUE(compiled) << compiled.error().message;
            EXPECT_TRUE(compiled.value().Validate("caller"));
            const auto invalid = compiled.value().Validate(rad::JsonObject{});
            ASSERT_FALSE(invalid);
            ASSERT_EQ(invalid.errors.size(), 1);
            EXPECT_EQ(invalid.errors[0].schemaPath, "/const");
            EXPECT_EQ(invalid.errors[0].schemaUri, options.documents[0].uri);
        }
        const auto local = rad::ParseJson(
            std::format(R"json({{"$id": "{}", "type": "object",
                               "properties": {{"value": {{"$ref": "{}"}}}}}})json",
                        draft.uri, draft.uri));
        ASSERT_TRUE(local);
        const auto compiled = rad::JsonSchema::Compile(local.value(), {.dialect = draft.dialect});
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(
            rad::JsonObject{{"value", rad::JsonObject{{"type", 7}}}}));

        if (draft.dialect != rad::JsonSchemaDialect::Draft7)
        {
            rad::JsonSchemaCompileOptions options{.dialect = draft.dialect};
            const auto uri = std::string(draft.uri.substr(0, draft.uri.rfind('/'))) +
                             "/meta/validation";
            options.documents = {
                {uri, rad::JsonObject{{"const", rad::JsonObject{{"type", 42}}}}},
            };
            const auto incomplete = rad::JsonSchema::Compile(root, options);
            ASSERT_FALSE(incomplete);
            EXPECT_EQ(incomplete.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
            EXPECT_EQ(incomplete.error().schemaUri, draft.uri);
            options.documents[0].schema.as_object()["$defs"] =
                rad::JsonObject{{"stringArray", true}};
            const auto overridden = rad::JsonSchema::Compile(root, options);
            ASSERT_TRUE(overridden) << overridden.error().message;
            EXPECT_TRUE(overridden.value().Validate(rad::JsonObject{{"type", 42}}));
        }
    }
    const auto mismatched = rad::JsonSchema::Compile(
        rad::JsonObject{{"$ref", "http://json-schema.org/draft-07/schema"}},
        {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_FALSE(mismatched);
    EXPECT_EQ(mismatched.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    EXPECT_EQ(mismatched.error().schemaPath, "/$schema");
    EXPECT_EQ(mismatched.error().schemaUri, "http://json-schema.org/draft-07/schema");
}

TEST(IO, JsonSchemaVocabularyDeclarations)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const std::string prefix = dialect == rad::JsonSchemaDialect::Draft2019_09
                                       ? "https://json-schema.org/draft/2019-09/vocab/"
                                       : "https://json-schema.org/draft/2020-12/vocab/";
        const auto core = prefix + "core";
        rad::JsonSchemaCompileOptions options{.dialect = dialect};
        options.retrievalUri = "https://example.com/root.json";
        const auto base = rad::JsonObject{
            {"$vocabulary", rad::JsonObject{{core, true}, {"urn:example:optional", false}}},
            {"type", "integer"}, {"minimum", 2},
        };
        const auto compiled = rad::JsonSchema::Compile(base, options);
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
            const auto invalid = rad::JsonSchema::Compile(schema, options);
            ASSERT_FALSE(invalid);
            EXPECT_EQ(invalid.error().code, testCase.code);
            EXPECT_EQ(invalid.error().schemaPath,
                      testCase.uri.starts_with("HTTPS:")
                          ? "/$vocabulary/HTTPS:~1~1example.com~1optional"
                          : "/$vocabulary/" + std::string(testCase.uri));
            EXPECT_EQ(invalid.error().schemaUri, options.retrievalUri);
        }
        auto format = base;
        const auto formatUri = prefix + (dialect == rad::JsonSchemaDialect::Draft2019_09
                                             ? "format" : "format-assertion");
        format.at("$vocabulary").as_object()[formatUri] = false;
        EXPECT_TRUE(rad::JsonSchema::Compile(format, {.dialect = dialect}));
        format.at("$vocabulary").as_object()[formatUri] = true;
        const auto requiredFormat = rad::JsonSchema::Compile(format, {.dialect = dialect});
        ASSERT_FALSE(requiredFormat);
        EXPECT_EQ(requiredFormat.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);

        for (const rad::JsonValue vocabulary :
             {rad::JsonValue(1), rad::JsonValue(rad::JsonObject{}),
              rad::JsonValue(rad::JsonObject{{core, false}})})
        {
            const auto invalid =
                rad::JsonSchema::Compile(rad::JsonObject{{"$vocabulary", vocabulary}}, options);
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
            EXPECT_EQ(invalid.error().schemaUri, options.retrievalUri);
        }
    }
    const auto ignored = rad::JsonSchema::Compile(rad::JsonObject{{"$vocabulary", 42}},
                                                  {.dialect = rad::JsonSchemaDialect::Draft7});
    ASSERT_TRUE(ignored) << ignored.error().message;
    EXPECT_TRUE(ignored.value().Validate(1));
}

TEST(IO, JsonSchemaCustomDialectSelection)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const std::string base = dialect == rad::JsonSchemaDialect::Draft2019_09
                                     ? "https://json-schema.org/draft/2019-09/"
                                     : "https://json-schema.org/draft/2020-12/";
        rad::JsonSchemaCompileOptions options;
        options.retrievalUri = "https://example.com/root.json";
        options.documents = {
            {"https://example.com/meta-a.json",
             rad::JsonObject{{"$id", "https://example.com/applicator"},
                             {"$schema", base + "schema"},
                             {"$vocabulary", rad::JsonObject{{base + "vocab/core", true},
                                                            {base + "vocab/applicator", true}}}}},
            {"https://example.com/meta-b.json",
             rad::JsonObject{{"$id", "https://example.com/validation"},
                             {"$schema", base + "schema"},
                             {"$vocabulary", rad::JsonObject{{base + "vocab/core", true},
                                                            {base + "vocab/validation", false}}}}},
        };
        const auto loose = rad::ParseJson(R"json({
            "$schema": "https://example.com/applicator",
            "type": 42, "enum": [], "required": false, "pattern": 1,
            "minimum": false, "minContains": "ignored", "uniqueItems": 1,
            "dependentRequired": 4, "properties": {"value": {"minimum": false}}
        })json");
        ASSERT_TRUE(loose);
        const auto inferred = rad::JsonSchema::Compile(loose.value(), options);
        ASSERT_TRUE(inferred) << inferred.error().schemaUri << inferred.error().schemaPath << ": "
                              << inferred.error().message;
        EXPECT_EQ(inferred.value().Dialect(), dialect);
        EXPECT_TRUE(inferred.value().Validate(rad::JsonObject{{"value", 1}}));
        options.dialect = dialect;
        EXPECT_TRUE(rad::JsonSchema::Compile(loose.value(), options));
        options.dialect = rad::JsonSchemaDialect::Auto;

        const auto strict = rad::ParseJson(R"json({
            "$schema": "https://example.com/validation", "type": "integer", "minimum": 1,
            "allOf": 5, "items": 5, "unevaluatedProperties": 5,
            "properties": {"bad": {"$id": 5, "$ref": "unregistered.json"}}
        })json");
        ASSERT_TRUE(strict);
        const auto compiled = rad::JsonSchema::Compile(strict.value(), options);
        ASSERT_TRUE(compiled) << compiled.error().schemaPath << ": " << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(1));
        EXPECT_FALSE(compiled.value().Validate(0));
        EXPECT_FALSE(compiled.value().Validate("invalid"));
        auto malformed = strict.value();
        malformed.as_object()["type"] = 42;
        const auto invalid = rad::JsonSchema::Compile(malformed, options);
        ASSERT_FALSE(invalid);
        EXPECT_EQ(invalid.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
        EXPECT_EQ(invalid.error().schemaPath, "/type");
        EXPECT_EQ(invalid.error().schemaUri, options.retrievalUri);

        auto pointed = strict.value();
        pointed.as_object()["$ref"] = "#/properties/bad";
        const auto badTarget = rad::JsonSchema::Compile(pointed, options);
        ASSERT_FALSE(badTarget);
        EXPECT_EQ(badTarget.error().schemaPath, "/properties/bad/$id");

        options.documents.push_back(
            {"https://example.com/defaults",
             rad::JsonObject{{"$schema", "https://example.com/validation"}}});
        const auto defaults = rad::JsonSchema::Compile(
            rad::JsonObject{{"$schema", "https://example.com/defaults"},
                            {"type", "object"},
                            {"properties", rad::JsonObject{{"value", false}}}},
            options);
        ASSERT_TRUE(defaults) << defaults.error().message;
        EXPECT_TRUE(defaults.value().Validate(rad::JsonObject{}));
        EXPECT_FALSE(defaults.value().Validate(rad::JsonObject{{"value", 1}}));
        EXPECT_FALSE(defaults.value().Validate(1));
    }
}

TEST(IO, JsonSchemaCustomDialectCandidateShapes)
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
        rad::JsonSchemaCompileOptions options;
        options.retrievalUri = "https://example.com/root.json";
        options.documents = {
            {"https://example.com/container",
             rad::JsonObject{{"$schema", base + "schema"}, {keyword, rad::JsonArray{meta}}}},
        };
        const rad::JsonValue schema = rad::JsonObject{
            {"$schema", "https://example.com/meta"}, {"type", "integer"}, {"allOf", 5}};
        const auto compiled = rad::JsonSchema::Compile(schema, options);
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(1));
        EXPECT_FALSE(compiled.value().Validate("invalid"));
        options.documents[0]
            .schema.as_object()
            .at(keyword)
            .as_array()[0]
            .as_object()["$vocabulary"] = rad::JsonObject{};
        const auto missingCore = rad::JsonSchema::Compile(schema, options);
        ASSERT_FALSE(missingCore);
        EXPECT_EQ(missingCore.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
        EXPECT_EQ(missingCore.error().schemaPath,
                  std::string("/") + keyword + "/0/$vocabulary/https:~1~1json-schema.org~1draft~1" +
                      (dialect == rad::JsonSchemaDialect::Draft2019_09 ? "2019-09" : "2020-12") +
                      "~1vocab~1core");
        EXPECT_EQ(missingCore.error().schemaUri, options.documents[0].uri);
        if (dialect == rad::JsonSchemaDialect::Draft2020_12)
        {
            options.documents[0].schema.as_object().erase("prefixItems");
            options.documents[0].schema.as_object()["items"] = rad::JsonArray{meta};
            const auto hidden = rad::JsonSchema::Compile(schema, options);
            ASSERT_FALSE(hidden);
            EXPECT_EQ(hidden.error().schemaPath, "/$schema");
            EXPECT_EQ(hidden.error().schemaUri, options.retrievalUri);
        }
    }
}

TEST(IO, JsonSchemaCustomDialectResourceProfiles)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        const std::string base = dialect == rad::JsonSchemaDialect::Draft2019_09
                                     ? "https://json-schema.org/draft/2019-09/"
                                     : "https://json-schema.org/draft/2020-12/";
        rad::JsonSchemaCompileOptions options;
        options.retrievalUri = "https://example.com/root.json";
        options.documents = {
            {"https://example.com/loose-meta",
             rad::JsonObject{{"$schema", base + "schema"},
                             {"$vocabulary", rad::JsonObject{{base + "vocab/core", true},
                                                            {base + "vocab/applicator", true}}}}},
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
        const auto compiled = rad::JsonSchema::Compile(schema.value(), options);
        ASSERT_TRUE(compiled) << compiled.error().schemaPath << ": " << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(rad::JsonObject{{"strict", 1}, {"loose", "ignored"}}));
        const auto invalid = compiled.value().Validate(rad::JsonObject{{"strict", "invalid"}});
        ASSERT_FALSE(invalid);
        ASSERT_EQ(invalid.errors.size(), 1);
        EXPECT_EQ(invalid.errors[0].instancePath, "/strict");
        EXPECT_EQ(invalid.errors[0].schemaPath, "/$defs/strict/type");
        EXPECT_EQ(invalid.errors[0].schemaUri, options.retrievalUri);

        auto nonResource = schema.value();
        nonResource.as_object().at("properties").as_object().at("strict") =
            rad::JsonObject{{"$schema", base + "schema"}, {"type", "integer"}};
        const auto badDeclaration = rad::JsonSchema::Compile(nonResource, options);
        ASSERT_FALSE(badDeclaration);
        EXPECT_EQ(badDeclaration.error().schemaPath, "/properties/strict/$schema");
        EXPECT_EQ(badDeclaration.error().schemaUri, options.retrievalUri);

        options.documents.push_back(
            {"https://example.com/foreign.json",
             rad::JsonObject{{"$schema", base + "schema"}, {"type", "integer"}}});
        auto referenced = schema.value();
        referenced.as_object()["$ref"] = "foreign.json";
        const auto external = rad::JsonSchema::Compile(referenced, options);
        ASSERT_TRUE(external) << external.error().message;
        EXPECT_TRUE(external.value().Validate(1));
        EXPECT_FALSE(external.value().Validate("invalid"));
        options.documents.back().schema.as_object()["$schema"] = "https://example.com/missing";
        const auto badExternal = rad::JsonSchema::Compile(referenced, options);
        ASSERT_FALSE(badExternal);
        EXPECT_EQ(badExternal.error().schemaPath, "/$schema");
        EXPECT_EQ(badExternal.error().schemaUri, options.documents.back().uri);
    }
}

TEST(IO, JsonSchemaCustomDialectUnevaluatedVocabulary)
{
    rad::JsonSchemaCompileOptions options;
    options.documents = {
        {"https://example.com/meta",
         rad::JsonObject{{"$schema", "https://json-schema.org/draft/2020-12/schema"},
                         {"$vocabulary", rad::JsonObject{
                             {"https://json-schema.org/draft/2020-12/vocab/core", true},
                             {"https://json-schema.org/draft/2020-12/vocab/unevaluated", false}}}}},
    };
    const auto compiled =
        rad::JsonSchema::Compile(rad::JsonObject{{"$schema", "https://example.com/meta"},
                                                 {"properties", rad::JsonObject{{"value", true}}},
                                                 {"prefixItems", rad::JsonArray{true}},
                                                 {"unevaluatedProperties", false},
                                                 {"unevaluatedItems", false}},
                                 options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    EXPECT_TRUE(compiled.value().Validate(rad::JsonObject{}));
    EXPECT_TRUE(compiled.value().Validate(rad::JsonArray{}));
    EXPECT_FALSE(compiled.value().Validate(rad::JsonObject{{"value", 1}}));
    EXPECT_FALSE(compiled.value().Validate(rad::JsonArray{1}));
}

TEST(IO, JsonSchemaCustomDialectDiagnostics)
{
    constexpr auto dialect = rad::JsonSchemaDialect::Draft2020_12;
    rad::JsonSchemaCompileOptions options;
    options.retrievalUri = "https://example.com/root.json";
    const auto root = rad::JsonObject{{"$schema", "https://example.com/meta"}};
    const auto missing = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().schemaPath, "/$schema");
    EXPECT_EQ(missing.error().schemaUri, options.retrievalUri);
    options.documents = {
        {"https://example.com/meta",
         rad::JsonObject{{"$schema", "https://example.com/other"}}},
        {"https://example.com/other",
         rad::JsonObject{{"$schema", "https://example.com/meta"}}},
    };
    const auto cyclic = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(cyclic);
    EXPECT_EQ(cyclic.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    EXPECT_EQ(cyclic.error().schemaPath, "/$schema");
    options.documents = {
        {"https://example.com/meta", rad::JsonObject{
            {"$schema", "https://json-schema.org/draft/2020-12/schema"},
            {"$vocabulary", rad::JsonObject{
                {"https://json-schema.org/draft/2020-12/vocab/core", true},
                {"urn:example:required", true}}},
        }},
    };
    const auto unsupported = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(unsupported);
    EXPECT_EQ(unsupported.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    EXPECT_EQ(unsupported.error().schemaPath, "/$vocabulary/urn:example:required");
    EXPECT_EQ(unsupported.error().schemaUri, options.documents[0].uri);
    options.documents[0].schema.as_object().at("$vocabulary").as_object()
        ["urn:example:required"] = false;
    EXPECT_TRUE(rad::JsonSchema::Compile(root, options));
    options.dialect = rad::JsonSchemaDialect::Draft2019_09;
    const auto mismatched = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(mismatched);
    EXPECT_EQ(mismatched.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    options.dialect = dialect;
    EXPECT_TRUE(rad::JsonSchema::Compile(root, options));
}
