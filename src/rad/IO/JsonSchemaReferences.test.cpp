#include <rad/IO/JsonSchema.h>
#include <rad/IO/JsonSchema.test.h>

#include <gtest/gtest.h>

#include <format>
#include <string>
#include <string_view>
#include <cstddef>
#include <utility>

using rad::test::MakeJsonDocument;

TEST(JsonSchemaReferences, EvaluationCycles)
{
    struct Case
    {
        std::string_view schema;
        std::string_view data;
        std::string_view instancePath;
        std::string_view schemaPath;
        std::string_view speculativeSchemaPath;
    };
    constexpr Case cases[] = {
        {R"json({"$ref": "#"})json", "1", "", ""},
        {R"json({"definitions": {"a": {"$ref": "#/definitions/b"},
                                "b": {"$ref": "#/definitions/a"}},
                 "$ref": "#/definitions/a"})json",
         "1", "", "/definitions/a"},
        {R"json({"allOf": [{"$ref": "#"}]})json", "1", "", ""},
        {R"json({"anyOf": [{"$ref": "#"}, true]})json", "1", "", "", "/anyOf/0"},
        {R"json({"oneOf": [{"$ref": "#"}, true]})json", "1", "", "", "/oneOf/0"},
        {R"json({"not": {"$ref": "#"}})json", "1", "", "", "/not"},
        {R"json({"if": {"$ref": "#"}, "then": true, "else": true})json", "1", "", "", "/if"},
        {R"json({"if": true, "then": {"$ref": "#"}})json", "1", "", ""},
        {R"json({"if": false, "else": {"$ref": "#"}})json", "1", "", ""},
        {R"json({"propertyNames": {"$ref": "#/propertyNames"}})json",
         R"json({"name": 1})json", "/name", "/propertyNames"},
        {R"json({"items": {"$ref": "#/items"}})json", "[1]", "/0", "/items"},
        {R"json({"contains": {"$ref": "#/contains"}})json", "[1]", "/0", "/contains"},
    };
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const std::string rootUri = "https://example.com/root.json";
        rad::JsonSchemaCompileOptions compileOptions{.dialect = dialect};
        for (const auto& testCase : cases)
        {
            SCOPED_TRACE(testCase.schema);
            const auto schema = rad::ParseJson(testCase.schema);
            const auto data = rad::ParseJson(testCase.data);
            ASSERT_TRUE(schema);
            ASSERT_TRUE(data);
            const auto compiled =
                rad::JsonSchema::Compile(MakeJsonDocument(schema.value(), rootUri), compileOptions);
            ASSERT_TRUE(compiled) << compiled.error().message;
            for (const std::size_t maxErrors : {0U, 1U, 2U, 64U})
            {
                SCOPED_TRACE(maxErrors);
                rad::JsonSchemaValidationOptions options;
                options.maxErrors = maxErrors;
                const auto result = compiled.value().Validate(data.value(), options);
                EXPECT_FALSE(result);
                EXPECT_EQ(result.status, rad::JsonSchemaValidationStatus::EvaluationCycleDetected);
                ASSERT_EQ(result.errors.size(), 1);
                EXPECT_EQ(result.errors[0].instancePath, testCase.instancePath);
                EXPECT_EQ(result.errors[0].schemaUri, rootUri);
                EXPECT_EQ(result.errors[0].schemaPath,
                          maxErrors > 1 && !testCase.speculativeSchemaPath.empty()
                              ? testCase.speculativeSchemaPath
                              : testCase.schemaPath);
                EXPECT_EQ(result.errors[0].message,
                          "schema evaluation cycle detected without instance progress");
            }
        }

        const auto compiled = rad::JsonSchema::Compile(
            MakeJsonDocument(rad::JsonObject{{"$ref", "#"}}, rootUri), compileOptions);
        ASSERT_TRUE(compiled);
        rad::JsonSchemaValidationOptions options;
        options.maxEvaluationDepth = 0;
        const auto limited = compiled.value().Validate(1, options);
        EXPECT_EQ(limited.status, rad::JsonSchemaValidationStatus::DepthLimitExceeded);
    }
}

TEST(JsonSchemaReferences, EvaluationCycleErrorLimits)
{
    struct Case
    {
        std::string_view schema;
        std::string_view schemaPath;
    };
    constexpr Case cases[] = {
        {R"json({"allOf": [false, {"$ref": "#"}]})json", ""},
        {R"json({"definitions": {"cycle": {"$ref": "#/definitions/cycle"}},
                 "allOf": [false, {"anyOf": [{"$ref": "#/definitions/cycle"}, true]}]})json",
         "/definitions/cycle"},
        {R"json({"definitions": {"cycle": {"$ref": "#/definitions/cycle"}},
                 "allOf": [false, {"oneOf": [{"$ref": "#/definitions/cycle"}, true]}]})json",
         "/definitions/cycle"},
        {R"json({"definitions": {"cycle": {"$ref": "#/definitions/cycle"}},
                 "allOf": [false, {"not": {"$ref": "#/definitions/cycle"}}]})json",
         "/definitions/cycle"},
        {R"json({"definitions": {"cycle": {"$ref": "#/definitions/cycle"}},
                 "allOf": [false, {"if": {"$ref": "#/definitions/cycle"}, "then": true}]})json",
         "/definitions/cycle"},
    };
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        for (const auto& testCase : cases)
        {
            SCOPED_TRACE(testCase.schema);
            const auto schema = rad::ParseJson(testCase.schema);
            ASSERT_TRUE(schema);
            const auto compiled =
                rad::JsonSchema::Compile(MakeJsonDocument(schema.value()), {.dialect = dialect});
            ASSERT_TRUE(compiled) << compiled.error().message;
            for (const std::size_t maxErrors : {0U, 1U, 2U, 64U})
            {
                SCOPED_TRACE(maxErrors);
                rad::JsonSchemaValidationOptions options;
                options.maxErrors = maxErrors;
                const auto result = compiled.value().Validate(1, options);
                EXPECT_FALSE(result);
                EXPECT_EQ(result.status,
                          maxErrors <= 1
                              ? rad::JsonSchemaValidationStatus::MaxErrorsReached
                              : rad::JsonSchemaValidationStatus::EvaluationCycleDetected);
                ASSERT_EQ(result.errors.size(), maxErrors <= 1 ? 1 : 2);
                EXPECT_EQ(result.errors[0].schemaPath, "/allOf/0");
                if (maxErrors > 1)
                {
                    EXPECT_EQ(result.errors.back().instancePath, "");
                    EXPECT_EQ(result.errors.back().schemaPath, testCase.schemaPath);
                    EXPECT_EQ(result.errors.back().message,
                              "schema evaluation cycle detected without instance progress");
                }
            }
        }
    }
}

TEST(JsonSchemaReferences, EvaluationCycleProgressAndSiblings)
{
    struct Case
    {
        std::string_view schema;
        std::string_view data;
        bool valid = true;
    };
    constexpr Case cases[] = {
        {R"json({"properties": {"left": {"$ref": "#"}, "right": {"$ref": "#"}},
                 "items": {"$ref": "#"}})json",
         R"json({"left": [1, 1, {"right": 2}], "right": {"left": 2}})json"},
        {R"json({"definitions": {"value": {"type": "integer"}},
                 "allOf": [{"$ref": "#/definitions/value"}, {"$ref": "#/definitions/value"}],
                 "anyOf": [{"$ref": "#/definitions/value"}, {"$ref": "#/definitions/value"}],
                 "oneOf": [{"$ref": "#/definitions/value"}, {"type": "string"}]})json",
         "1"},
        {R"json({"propertyNames": {"$ref": "#"}})json", R"json({"name": 1})json"},
        {R"json({"anyOf": [{"allOf": [false, {"$ref": "#"}]}, true]})json", "1"},
        {R"json({"not": {"allOf": [false, {"$ref": "#"}]}})json", "1"},
        {R"json({"if": {"allOf": [false, {"$ref": "#"}]},
                 "then": false, "else": true})json",
         "1"},
        {R"json({"allOf": [false, {"anyOf": [{"$ref": "#"}, true]}]})json", "1", false},
        {R"json({"allOf": [false, {"oneOf": [{"$ref": "#"}, true]}]})json", "1", false},
        {R"json({"allOf": [false, {"not": {"$ref": "#"}}]})json", "1", false},
        {R"json({"allOf": [false, {"if": {"$ref": "#"}, "then": true}]})json", "1", false},
    };
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        for (const auto& testCase : cases)
        {
            SCOPED_TRACE(testCase.schema);
            const auto schema = rad::ParseJson(testCase.schema);
            const auto data = rad::ParseJson(testCase.data);
            ASSERT_TRUE(schema);
            ASSERT_TRUE(data);
            const auto compiled =
                rad::JsonSchema::Compile(MakeJsonDocument(schema.value()), {.dialect = dialect});
            ASSERT_TRUE(compiled) << compiled.error().message;
            for (std::size_t iteration = 0; iteration < 2; ++iteration)
            {
                const auto result = compiled.value().Validate(data.value());
                EXPECT_EQ(static_cast<bool>(result), testCase.valid);
                EXPECT_EQ(result.status, rad::JsonSchemaValidationStatus::Completed);
                if (testCase.valid)
                {
                    EXPECT_TRUE(result.errors.empty());
                }
                else
                {
                    ASSERT_EQ(result.errors.size(), 1);
                    EXPECT_EQ(result.errors[0].schemaPath, "/allOf/0");
                }
            }
        }
    }
}

TEST(JsonSchemaReferences, EvaluationCycleDynamicScopes)
{
    std::string rootUri = "https://example.com/root.json";
    rad::JsonSchemaCompileOptions options{.dialect = rad::JsonSchemaDialect::Draft2020_12};
    options.documents = {
        MakeJsonDocument(rad::JsonObject{{"$dynamicAnchor", "condition"}},
                         "https://example.com/fallback.json"),
        MakeJsonDocument(rad::ParseJson(R"json({"$ref": "root.json",
             "$defs": {"condition": {"$dynamicAnchor": "condition", "not": true}}})json")
                             .value(),
                         "https://example.com/bridge.json"),
    };
    const auto schema = rad::ParseJson(
        R"json({"if": {"$dynamicRef": "fallback.json#condition"},
                 "then": {"$ref": "bridge.json"}, "else": true})json");
    ASSERT_TRUE(schema);
    const auto compiled =
        rad::JsonSchema::Compile(MakeJsonDocument(schema.value(), rootUri), options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    const auto result = compiled.value().Validate(1);
    EXPECT_TRUE(result);
    EXPECT_EQ(result.status, rad::JsonSchemaValidationStatus::Completed);
    EXPECT_TRUE(result.errors.empty());

    rootUri = "https://example.com/a.json";
    options.documents = {
        MakeJsonDocument(rad::JsonObject{{"$dynamicAnchor", "b"}, {"$dynamicRef", "a.json#a"}},
                         "https://example.com/b.json"),
    };
    const auto cyclic = rad::JsonSchema::Compile(
        MakeJsonDocument(rad::JsonObject{{"$dynamicAnchor", "a"}, {"$dynamicRef", "b.json#b"}},
                         rootUri),
        options);
    ASSERT_TRUE(cyclic) << cyclic.error().message;
    rad::JsonSchemaValidationOptions limits;
    limits.maxEvaluationDepth = 3;
    const auto cycle = cyclic.value().Validate(1, limits);
    EXPECT_FALSE(cycle);
    EXPECT_EQ(cycle.status, rad::JsonSchemaValidationStatus::EvaluationCycleDetected);
    ASSERT_EQ(cycle.errors.size(), 1);
    EXPECT_EQ(cycle.errors[0].instancePath, "");
    EXPECT_EQ(cycle.errors[0].schemaUri, "https://example.com/b.json");
    EXPECT_EQ(cycle.errors[0].schemaPath, "");
}

TEST(JsonSchemaReferences, EvaluationCycleRecursiveScopes)
{
    std::string rootUri = "https://example.com/entry.json";
    rad::JsonSchemaCompileOptions options{.dialect = rad::JsonSchemaDialect::Draft2019_09};
    options.documents = {
        MakeJsonDocument(rad::ParseJson(R"json({"$recursiveAnchor": true, "$defs": {"entry": {
             "if": {"$ref": "probe.json"}, "then": {"$ref": "barrier.json"},
             "else": true}}})json")
                             .value(),
                         "https://example.com/root.json"),
        MakeJsonDocument(rad::JsonObject{{"$recursiveAnchor", true}, {"$recursiveRef", "#"}},
                         "https://example.com/probe.json"),
        MakeJsonDocument(rad::JsonObject{{"$ref", "bridge.json#/$defs/entry"}},
                         "https://example.com/barrier.json"),
        MakeJsonDocument(rad::ParseJson(R"json({"$recursiveAnchor": true, "not": true,
             "$defs": {"entry": {"$ref": "root.json#/$defs/entry"}}})json")
                             .value(),
                         "https://example.com/bridge.json"),
    };
    const auto compiled = rad::JsonSchema::Compile(
        MakeJsonDocument(rad::JsonObject{{"$ref", "root.json#/$defs/entry"}}, rootUri), options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    const auto result = compiled.value().Validate(1);
    EXPECT_TRUE(result);
    EXPECT_EQ(result.status, rad::JsonSchemaValidationStatus::Completed);
    EXPECT_TRUE(result.errors.empty());

    rootUri = "https://example.com/a.json";
    options.documents = {
        MakeJsonDocument(rad::JsonObject{{"$recursiveAnchor", true}, {"$recursiveRef", "#"}},
                         "https://example.com/b.json"),
    };
    const auto cyclic = rad::JsonSchema::Compile(
        MakeJsonDocument(rad::JsonObject{{"$recursiveAnchor", true}, {"$ref", "b.json"}}, rootUri),
        options);
    ASSERT_TRUE(cyclic) << cyclic.error().message;
    rad::JsonSchemaValidationOptions limits;
    limits.maxEvaluationDepth = 2;
    const auto cycle = cyclic.value().Validate(1, limits);
    EXPECT_FALSE(cycle);
    EXPECT_EQ(cycle.status, rad::JsonSchemaValidationStatus::EvaluationCycleDetected);
    ASSERT_EQ(cycle.errors.size(), 1);
    EXPECT_EQ(cycle.errors[0].instancePath, "");
    EXPECT_EQ(cycle.errors[0].schemaUri, "https://example.com/a.json");
    EXPECT_EQ(cycle.errors[0].schemaPath, "");
}

TEST(JsonSchemaReferences, DiagnosticsAndLimits)
{
    const auto schema = rad::ParseJson(
        R"json({"$defs": {"node": {"type": "object", "properties":
                 {"next": {"$ref": "#/$defs/node"}}}},
                 "$ref": "#/$defs/node"})json");
    ASSERT_TRUE(schema);
    const auto compiled = rad::JsonSchema::Compile(
        MakeJsonDocument(schema.value()), {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_TRUE(compiled) << compiled.error().message;
    const auto invalid = compiled.value().Validate(rad::JsonObject{
        {"next", rad::JsonObject{{"next", 12}}}});
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.errors[0].instancePath, "/next/next");
    EXPECT_EQ(invalid.errors[0].schemaPath, "/$defs/node/type");

    rad::JsonSchemaValidationOptions options;
    options.maxEvaluationDepth = 2;
    const auto limited = compiled.value().Validate(rad::JsonObject{
        {"next", rad::JsonObject{{"next", rad::JsonObject{}}}}}, options);
    ASSERT_FALSE(limited);
    ASSERT_EQ(limited.errors.size(), 1);
    EXPECT_EQ(limited.status, rad::JsonSchemaValidationStatus::DepthLimitExceeded);
    EXPECT_NE(limited.errors.back().message.find("maximum validation depth"),
              std::string::npos);

    const auto selfReference = rad::ParseJson(R"json({"$ref": "#"})json");
    ASSERT_TRUE(selfReference);
    const auto selfCompiled = rad::JsonSchema::Compile(
        MakeJsonDocument(selfReference.value()), {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_TRUE(selfCompiled);
    options.maxErrors = 1;
    const auto selfResult = selfCompiled.value().Validate(1, options);
    EXPECT_FALSE(selfResult);
    ASSERT_EQ(selfResult.errors.size(), 1);
    EXPECT_EQ(selfResult.status, rad::JsonSchemaValidationStatus::EvaluationCycleDetected);
    EXPECT_EQ(selfResult.errors.back().schemaPath, "");
    EXPECT_EQ(selfResult.errors.back().message,
              "schema evaluation cycle detected without instance progress");
}

TEST(JsonSchemaReferences, ResourceLimitsAndCopies)
{
    const auto nestedSource = rad::ParseJson(
        R"json({"$defs": {"inner": {"$id": "sub.json", "$ref": "#"}},
                 "$ref": "#/$defs/inner"})json");
    ASSERT_TRUE(nestedSource);
    const auto nestedCompiled = rad::JsonSchema::Compile(
        MakeJsonDocument(nestedSource.value()), {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_TRUE(nestedCompiled) << nestedCompiled.error().message;
    rad::JsonSchemaValidationOptions options;
    options.maxEvaluationDepth = 2;
    const auto recursive = nestedCompiled.value().Validate(1, options);
    ASSERT_FALSE(recursive);
    ASSERT_EQ(recursive.errors.size(), 1);
    EXPECT_EQ(recursive.status, rad::JsonSchemaValidationStatus::EvaluationCycleDetected);
    EXPECT_EQ(recursive.errors.back().schemaPath, "/$defs/inner");
    EXPECT_EQ(recursive.errors.back().message,
              "schema evaluation cycle detected without instance progress");

    const auto makeSchema = []
    {
        const auto schema = rad::ParseJson(
            R"json({"$defs": {"inner": {"$id": "sub.json", "type": "string"}},
                     "$ref": "sub.json"})json");
        return rad::JsonSchema::Compile(MakeJsonDocument(schema.value()),
                                        {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    };
    const auto compiled = makeSchema();
    ASSERT_TRUE(compiled) << compiled.error().message;
    auto copied = compiled.value();
    auto moved = std::move(copied);
    const auto invalid = moved.Validate(1);
    ASSERT_FALSE(invalid);
    ASSERT_EQ(invalid.errors.size(), 1);
    EXPECT_EQ(invalid.errors[0].schemaPath, "/$defs/inner/type");
}

TEST(JsonSchemaReferences, ResourceIdentifierDiagnostics)
{
    struct Case
    {
        std::string_view schema;
        std::string_view path;
    };
    constexpr Case cases[] = {
        {R"json({"$id": 42})json", "/$id"},
        {R"json({"$id": "https://example.com/%ZZ"})json", "/$id"},
        {R"json({"$id": "#name"})json", "/$id"},
        {R"json({"$anchor": 42})json", "/$anchor"},
        {R"json({"$anchor": "1invalid"})json", "/$anchor"},
        {R"json({"$defs": {"a": {"$anchor": "same"}, "b": {"$anchor": "same"}}})json",
         "/$defs/b/$anchor"},
        {R"json({"$defs": {"a": {"$id": "https://example.com/a"},
                           "b": {"$id": "https://EXAMPLE.COM/%61"}}})json",
         "/$defs/b/$id"},
    };
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        for (const auto& testCase : cases)
        {
            SCOPED_TRACE(testCase.schema);
            const auto schema = rad::ParseJson(testCase.schema);
            ASSERT_TRUE(schema);
            const auto compiled =
                rad::JsonSchema::Compile(MakeJsonDocument(schema.value()), {.dialect = dialect});
            ASSERT_FALSE(compiled);
            EXPECT_EQ(compiled.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
            EXPECT_EQ(compiled.error().schemaPath, testCase.path);
        }

        const auto deferred = rad::ParseJson(
            R"json({"allOf": [{"$ref": "child#target"}, {"$ref": "#/storage"}],
                     "storage": {"$id": "child", "$anchor": "target",
                                 "$defs": {"value": {"type": "string"}},
                                 "$ref": "#/$defs/value"}})json");
        ASSERT_TRUE(deferred);
        const auto compiled =
            rad::JsonSchema::Compile(MakeJsonDocument(deferred.value()), {.dialect = dialect});
        ASSERT_TRUE(compiled) << compiled.error().schemaPath << ": " << compiled.error().message;
        const auto result = compiled.value().Validate(1);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.errors[0].schemaPath, "/storage/$defs/value/type");

        const auto annotations = rad::ParseJson(
            R"json({"const": {"$id": 42, "$anchor": "1invalid"},
                     "examples": [{"$id": "hidden", "$ref": "external"}]})json");
        ASSERT_TRUE(annotations);
        EXPECT_TRUE(
            rad::JsonSchema::Compile(MakeJsonDocument(annotations.value()), {.dialect = dialect}));
    }
    const auto ignored = rad::ParseJson(
        R"json({"$ref": "#/definitions/value", "$id": 42,
                 "definitions": {"value": true, "unused": {"$ref": "external.json"},
                                 "malformed": {"$ref": 42}}})json");
    ASSERT_TRUE(ignored);
    EXPECT_TRUE(rad::JsonSchema::Compile(MakeJsonDocument(ignored.value()),
                                         {.dialect = rad::JsonSchemaDialect::Draft7}));
}

TEST(JsonSchemaReferences, DiscoveryOrder)
{
    for (const auto dialect :
         {rad::JsonSchemaDialect::Draft2019_09, rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        for (const bool descendantFirst : {true, false})
        {
            SCOPED_TRACE(descendantFirst);
            constexpr std::string_view schemas[] = {
                R"json({"allOf": [{"$ref": "#/storage/$defs/use"}, {"$ref": "#/storage"}],
                         "storage": {"$id": "child",
                                     "$defs": {"use": {"$ref": "#/$defs/value"},
                                               "value": {"type": "integer"}}}})json",
                R"json({"allOf": [{"$ref": "#/storage/$defs/use"}, {"$ref": "#/entry"},
                                 {"$ref": "child/use.json#use"}],
                         "entry": {"$ref": "#/storage"},
                         "storage": {"$id": "child/",
                                     "$defs": {"use": {"$id": "use.json", "$anchor": "use",
                                                       "$ref": "value.json"},
                                               "value": {"$id": "value.json",
                                                         "type": "integer"}}}})json",
            };
            for (const auto text : schemas)
            {
                SCOPED_TRACE(text);
                const auto schema = rad::ParseJson(text);
                ASSERT_TRUE(schema);
                auto value = schema.value();
                if (!descendantFirst)
                {
                    auto& branches = value.as_object().at("allOf").as_array();
                    std::swap(branches[0], branches[1]);
                }
                const auto compiled =
                    rad::JsonSchema::Compile(MakeJsonDocument(value), {.dialect = dialect});
                ASSERT_TRUE(compiled) << compiled.error().schemaPath << ": "
                                      << compiled.error().message;
                EXPECT_TRUE(compiled.value().Validate(1));
                const auto invalid = compiled.value().Validate("invalid");
                ASSERT_FALSE(invalid);
                rad::JsonSchemaValidationOptions options;
                options.maxErrors = 1;
                const auto limited = compiled.value().Validate("invalid", options);
                ASSERT_FALSE(limited);
                ASSERT_EQ(limited.errors.size(), 1);
                EXPECT_EQ(limited.errors[0].schemaPath, "/storage/$defs/value/type");
            }
        }
    }
}

TEST(JsonSchemaReferences, RejectsInvalidReferences)
{
    struct Case
    {
        std::string_view reference;
        rad::JsonSchemaCompileErrorCode code;
    };
    constexpr Case cases[] = {
        {R"json(7)json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("#/missing")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("#/a~2b")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("#/%ZZ")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("#/list/01")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("#/list/-")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("#/list/1")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json("other.json#/$defs/item")json",
         rad::JsonSchemaCompileErrorCode::UnsupportedFeature},
        {R"json("#named")json", rad::JsonSchemaCompileErrorCode::InvalidSchema},
    };
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.reference);
        const auto schema = rad::ParseJson(
            std::format(R"json({{"list": [{{"type": "string"}}], "$ref": {}}})json",
                        testCase.reference));
        ASSERT_TRUE(schema);
        const auto compiled = rad::JsonSchema::Compile(
            MakeJsonDocument(schema.value()), {.dialect = rad::JsonSchemaDialect::Draft2020_12});
        ASSERT_FALSE(compiled);
        EXPECT_EQ(compiled.error().code, testCase.code);
        EXPECT_EQ(compiled.error().schemaPath, "/$ref");
    }

    const auto nonSchemaTarget = rad::ParseJson(
        R"json({"list": [{"type": "string"}], "$ref": "#/list/0/type"})json");
    ASSERT_TRUE(nonSchemaTarget);
    const auto invalidTarget =
        rad::JsonSchema::Compile(MakeJsonDocument(nonSchemaTarget.value()),
                                 {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_FALSE(invalidTarget);
    EXPECT_EQ(invalidTarget.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    EXPECT_EQ(invalidTarget.error().schemaPath, "/list/0/type");
}

TEST(JsonSchemaReferences, RootDocument)
{
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        rad::JsonSchemaCompileOptions options{.dialect = dialect};
        const auto document = MakeJsonDocument(
            rad::JsonObject{
                {"$id", "canonical/root.json"},
                {"type", "object"},
                {"properties",
                 rad::JsonObject{{"number", rad::JsonObject{{"type", "integer"}}},
                                 {"value", rad::JsonObject{{"$ref", "../child.json"}}}}},
            },
            "https://example.com/schemas/root.json");
        options.documents.push_back(
            MakeJsonDocument(rad::JsonObject{{"$ref", "root.json#/properties/number"}},
                             "https://example.com/schemas/child.json"));
        const auto compiled = rad::JsonSchema::Compile(document, options);
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(rad::JsonObject{{"value", 1}}));
        const auto invalid = compiled.value().Validate(rad::JsonObject{{"value", "invalid"}});
        ASSERT_FALSE(invalid);
        ASSERT_EQ(invalid.errors.size(), 1);
        EXPECT_EQ(invalid.errors[0].instancePath, "/value");
        EXPECT_EQ(invalid.errors[0].schemaUri, document->Uri());
        EXPECT_EQ(invalid.errors[0].schemaPath, "/properties/number/type");

        const auto anonymous = rad::JsonSchema::Compile(
            MakeJsonDocument(rad::JsonObject{{"type", "integer"}}), options);
        ASSERT_TRUE(anonymous) << anonymous.error().message;
        const auto anonymousInvalid = anonymous.value().Validate("invalid");
        ASSERT_FALSE(anonymousInvalid);
        ASSERT_EQ(anonymousInvalid.errors.size(), 1);
        EXPECT_TRUE(anonymousInvalid.errors[0].schemaUri.empty());
        EXPECT_EQ(anonymousInvalid.errors[0].schemaPath, "/type");

        options.documents[0] = MakeJsonDocument(options.documents[0]->Root(), document->Uri());
        const auto duplicate = rad::JsonSchema::Compile(document, options);
        ASSERT_FALSE(duplicate);
        EXPECT_EQ(duplicate.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
        EXPECT_EQ(duplicate.error().schemaUri, document->Uri());

        options.documents[0] = MakeJsonDocument(options.documents[0]->Root());
        const auto unregistered = rad::JsonSchema::Compile(MakeJsonDocument(true), options);
        ASSERT_FALSE(unregistered);
        EXPECT_EQ(unregistered.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    }
}

TEST(JsonSchemaReferences, RetainsSharedDocuments)
{
    auto root = rad::JsonDocument::Parse(R"json({"$ref": "child.json"})json");
    auto child = rad::JsonDocument::Parse(R"json({"type": "integer"})json");
    ASSERT_TRUE(root) << root.error().message();
    ASSERT_TRUE(child) << child.error().message();
    root.value()->SetUri("https://example.com/root.json");
    child.value()->SetUri("https://example.com/child.json");
    rad::JsonSchemaCompileOptions options{.dialect = rad::JsonSchemaDialect::Draft2020_12};
    options.documents = {child.value()};
    const auto compiled = rad::JsonSchema::Compile(root.value(), options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    root.value().reset();
    child.value().reset();
    options.documents.clear();

    EXPECT_TRUE(compiled.value().Validate(1));
    const auto invalid = compiled.value().Validate("invalid");
    ASSERT_FALSE(invalid);
    ASSERT_EQ(invalid.errors.size(), 1);
    EXPECT_EQ(invalid.errors[0].schemaUri, "https://example.com/child.json");
    EXPECT_EQ(invalid.errors[0].schemaPath, "/type");
}

TEST(JsonSchemaReferences, NullDocuments)
{
    rad::JsonSchemaCompileOptions options{.dialect = rad::JsonSchemaDialect::Draft2020_12};
    const auto missingRoot = rad::JsonSchema::Compile(rad::Ref<const rad::JsonDocument>{}, options);
    ASSERT_FALSE(missingRoot);
    EXPECT_EQ(missingRoot.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);

    const auto root = MakeJsonDocument(true, "https://example.com/root.json");
    options.documents.emplace_back();
    const auto missingDocument = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(missingDocument);
    EXPECT_EQ(missingDocument.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    EXPECT_EQ(missingDocument.error().schemaUri, root->Uri());
}

TEST(JsonSchemaReferences, DocumentRegistry)
{
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const auto makeSchema = [&]
        {
            const std::string rootUri = "https://example.com/root.json";
            rad::JsonSchemaCompileOptions options{.dialect = dialect};
            options.documents = {
                MakeJsonDocument(
                    rad::ParseJson(R"json({"$id": "canonical/a.json", "type": "object",
                     "properties": {"child": {"$ref": "../b.json#/properties/value"}}})json")
                        .value(),
                    "https://example.com/a.json"),
                MakeJsonDocument(
                    rad::ParseJson(R"json({"properties": {"value": {"type": "integer"},
                     "embedded": {"$id": "embedded.json", "type": "string"}}})json")
                        .value(),
                    "https://example.com/b.json"),
            };
            auto root = rad::JsonObject{{"$ref", "canonical/a.json"}};
            if (dialect != rad::JsonSchemaDialect::Draft7)
            {
                root["unevaluatedProperties"] = false;
            }
            return rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options);
        };
        const auto compiled = makeSchema();
        ASSERT_TRUE(compiled) << compiled.error().schemaUri << compiled.error().schemaPath << ": "
                              << compiled.error().message;
        auto copied = compiled.value();
        auto moved = std::move(copied);
        EXPECT_TRUE(moved.Validate(rad::JsonObject{{"child", 1}}));
        rad::JsonSchemaValidationOptions diagnosticLimit;
        diagnosticLimit.maxErrors = 1;
        const auto invalid = moved.Validate(rad::JsonObject{{"child", "invalid"}}, diagnosticLimit);
        ASSERT_FALSE(invalid);
        ASSERT_EQ(invalid.errors.size(), 1);
        EXPECT_EQ(invalid.errors[0].instancePath, "/child");
        EXPECT_EQ(invalid.errors[0].schemaPath, "/properties/value/type");
        EXPECT_EQ(invalid.errors[0].schemaUri, "https://example.com/b.json");
        if (dialect != rad::JsonSchemaDialect::Draft7)
        {
            EXPECT_FALSE(moved.Validate(rad::JsonObject{{"child", 1}, {"extra", 1}}));
        }

        const std::string rootUri = "https://example.com/root.json";
        rad::JsonSchemaCompileOptions options{.dialect = dialect};
        options.documents = {
            MakeJsonDocument(rad::ParseJson(R"json({"properties":
                 {"embedded": {"$id": "embedded.json", "type": "integer"}}})json")
                                 .value(),
                             "https://example.com/container.json"),
        };
        const auto embedded = rad::JsonSchema::Compile(
            MakeJsonDocument(rad::JsonObject{{"$ref", "embedded.json"}}, rootUri), options);
        ASSERT_TRUE(embedded) << embedded.error().message;
        EXPECT_TRUE(embedded.value().Validate(1));
        const auto failure = embedded.value().Validate("invalid");
        ASSERT_FALSE(failure);
        EXPECT_EQ(failure.errors[0].schemaUri, "https://example.com/container.json");
        EXPECT_EQ(failure.errors[0].schemaPath, "/properties/embedded/type");

        options.documents = {
            MakeJsonDocument(rad::ParseJson(R"json({"$id": "canonical.json",
                                   "properties": {"value": {"$anchor": "value",
                                                           "type": "integer"}}})json")
                                 .value(),
                             "https://example.com/alias.json"),
        };
        if (dialect == rad::JsonSchemaDialect::Draft7)
        {
            auto schema = options.documents[0]->Root();
            auto& value = schema.as_object().at("properties").as_object().at("value").as_object();
            value.erase("$anchor");
            value["$id"] = "#value";
            options.documents[0] = MakeJsonDocument(std::move(schema), options.documents[0]->Uri());
        }
        for (const auto* reference :
             {"alias.json#value", "canonical.json#value", "alias.json#/properties/value"})
        {
            const auto aliased = rad::JsonSchema::Compile(
                MakeJsonDocument(rad::JsonObject{{"$ref", reference}}, rootUri), options);
            ASSERT_TRUE(aliased) << aliased.error().message;
            const auto invalid = aliased.value().Validate("invalid");
            ASSERT_FALSE(invalid);
            EXPECT_EQ(invalid.errors[0].schemaUri, "https://example.com/alias.json");
            EXPECT_EQ(invalid.errors[0].schemaPath, "/properties/value/type");
        }

        options.documents = {
            MakeJsonDocument(rad::JsonObject{{"$ref", "b.json"}}, "https://example.com/a.json"),
            MakeJsonDocument(rad::JsonObject{{"$ref", "a.json"}}, "https://example.com/b.json"),
        };
        const auto cyclic = rad::JsonSchema::Compile(
            MakeJsonDocument(rad::JsonObject{{"$ref", "a.json"}}, rootUri), options);
        ASSERT_TRUE(cyclic) << cyclic.error().message;
        rad::JsonSchemaValidationOptions limits;
        limits.maxEvaluationDepth = 2;
        limits.maxErrors = 1;
        const auto exhausted = cyclic.value().Validate(1, limits);
        ASSERT_FALSE(exhausted);
        ASSERT_EQ(exhausted.errors.size(), 1);
        EXPECT_EQ(exhausted.status, rad::JsonSchemaValidationStatus::DepthLimitExceeded);
        EXPECT_EQ(exhausted.errors.back().schemaPath, "");
        EXPECT_EQ(exhausted.errors.back().schemaUri, "https://example.com/a.json");
        EXPECT_EQ(exhausted.errors.back().message, "maximum validation depth exceeded");

        const auto cycle = cyclic.value().Validate(1);
        ASSERT_FALSE(cycle);
        ASSERT_EQ(cycle.errors.size(), 1);
        EXPECT_EQ(cycle.status, rad::JsonSchemaValidationStatus::EvaluationCycleDetected);
        EXPECT_EQ(cycle.errors.back().schemaPath, "");
        EXPECT_EQ(cycle.errors.back().schemaUri, "https://example.com/a.json");
        EXPECT_EQ(cycle.errors.back().message,
                  "schema evaluation cycle detected without instance progress");

        options.documents = {
            MakeJsonDocument(rad::ParseJson(R"json({"type": "integer",
                 "properties": {"child": {"allOf": [true]}}})json")
                                 .value(),
                             "https://example.com/a.json"),
        };
        const auto branches = rad::ParseJson(R"json({"anyOf": [{"$ref": "a.json"}, true]})json");
        ASSERT_TRUE(branches);
        const auto limited =
            rad::JsonSchema::Compile(MakeJsonDocument(branches.value(), rootUri), options);
        ASSERT_TRUE(limited) << limited.error().message;
        limits.maxEvaluationDepth = 3;
        for (const std::size_t maxErrors : {0U, 1U, 2U})
        {
            limits.maxErrors = maxErrors;
            const auto result =
                limited.value().Validate(rad::JsonObject{{"child", rad::JsonObject{}}}, limits);
            EXPECT_TRUE(result);
            EXPECT_EQ(result.status, rad::JsonSchemaValidationStatus::Completed);
            EXPECT_TRUE(result.errors.empty());
        }
        limits.maxEvaluationDepth = 4;
        EXPECT_TRUE(
            limited.value().Validate(rad::JsonObject{{"child", rad::JsonObject{}}}, limits));
    }
}

TEST(JsonSchemaReferences, DocumentRegistryDiagnostics)
{
    constexpr auto dialect = rad::JsonSchemaDialect::Draft2020_12;
    std::string rootUri = "https://example.com/root.json";
    rad::JsonSchemaCompileOptions options{.dialect = dialect};
    const auto root = rad::JsonObject{{"$ref", "child.json"}};
    const auto missing = rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options);
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    EXPECT_EQ(missing.error().schemaPath, "/$ref");
    EXPECT_EQ(missing.error().schemaUri, rootUri);

    options.documents = {
        MakeJsonDocument(rad::JsonObject{{"type", 42}}, "https://example.com/child.json"),
    };
    const auto invalid = rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options);
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.error().schemaPath, "/type");
    EXPECT_EQ(invalid.error().schemaUri, "https://example.com/child.json");
    EXPECT_TRUE(rad::JsonSchema::Compile(MakeJsonDocument(rad::JsonObject{}, rootUri), options));

    options.documents[0] = MakeJsonDocument(
        rad::JsonObject{
            {"$schema", "http://json-schema.org/draft-07/schema"},
            {"properties", rad::JsonObject{{"child", true}}},
        },
        "https://example.com/child.json");
    const auto mixed = rad::JsonSchema::Compile(
        MakeJsonDocument(rad::JsonObject{{"$ref", "child.json#/properties/child"}}, rootUri),
        options);
    ASSERT_FALSE(mixed);
    EXPECT_EQ(mixed.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    EXPECT_EQ(mixed.error().schemaPath, "/$schema");
    EXPECT_EQ(mixed.error().schemaUri, "https://example.com/child.json");

    options.documents[0] = MakeJsonDocument(true, "https://example.com/child.json");
    options.documents.push_back(MakeJsonDocument(false, "https://EXAMPLE.COM/%63hild.json"));
    const auto duplicate = rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options);
    ASSERT_FALSE(duplicate);
    EXPECT_EQ(duplicate.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    options.documents.pop_back();
    const auto conflict = rad::JsonSchema::Compile(
        MakeJsonDocument(rad::JsonObject{{"$id", "https://example.com/child.json"}}, rootUri),
        options);
    ASSERT_FALSE(conflict);
    EXPECT_EQ(conflict.error().schemaPath, "/$id");
    EXPECT_EQ(conflict.error().schemaUri, rootUri);

    auto detected = rad::JsonObject{{"$schema", "https://json-schema.org/draft/2020-12/schema"},
                                    {"$ref", "child.json"}};
    options.dialect = rad::JsonSchemaDialect::Draft2020_12;
    EXPECT_TRUE(rad::JsonSchema::Compile(MakeJsonDocument(detected, rootUri), options));
    EXPECT_FALSE(rad::JsonSchema::Compile(MakeJsonDocument(detected)));
    options.dialect = dialect;
    rootUri = "relative.json";
    const auto relative = rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options);
    ASSERT_FALSE(relative);
    EXPECT_EQ(relative.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    rootUri = "https://example.com/root.json#fragment";
    EXPECT_FALSE(rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options));

    rootUri = "https://example.com/root.json";
    options.documents[0] =
        MakeJsonDocument(rad::JsonObject{{"$id", 42}}, "https://example.com/child.json");
    const auto ignored = rad::ParseJson(
        R"json({"$ref": "#/definitions/value",
                 "definitions": {"value": true, "unused": {"$ref": "child.json"}}})json");
    ASSERT_TRUE(ignored);
    options.dialect = rad::JsonSchemaDialect::Draft7;
    EXPECT_TRUE(rad::JsonSchema::Compile(MakeJsonDocument(ignored.value(), rootUri), options));
    options.dialect = dialect;
    const auto malformedId = rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options);
    ASSERT_FALSE(malformedId);
    EXPECT_EQ(malformedId.error().schemaPath, "/$id");
    EXPECT_EQ(malformedId.error().schemaUri, "https://example.com/child.json");
}

TEST(JsonSchemaReferences, RecursiveDiagnostics)
{
    constexpr auto dialect = rad::JsonSchemaDialect::Draft2019_09;
    struct Case
    {
        std::string_view schema;
        std::string_view path;
        rad::JsonSchemaCompileErrorCode code;
    };
    constexpr Case cases[] = {
        {R"json({"$recursiveAnchor": 1})json", "/$recursiveAnchor",
         rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json({"$recursiveRef": false})json", "/$recursiveRef",
         rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json({"$recursiveRef": "#/%ZZ"})json", "/$recursiveRef",
         rad::JsonSchemaCompileErrorCode::InvalidSchema},
        {R"json({"$recursiveRef": "#/$defs/value", "$defs": {"value": true}})json",
         "/$recursiveRef", rad::JsonSchemaCompileErrorCode::UnsupportedFeature},
        {R"json({"$recursiveRef": "other.json#"})json", "/$recursiveRef",
         rad::JsonSchemaCompileErrorCode::UnsupportedFeature},
        {R"json({"properties": {"child": {"$recursiveAnchor": "true"}}})json",
         "/properties/child/$recursiveAnchor", rad::JsonSchemaCompileErrorCode::InvalidSchema},
    };
    const std::string rootUri = "https://example.com/root.json";
    rad::JsonSchemaCompileOptions options{.dialect = dialect};
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto parsed = rad::ParseJson(testCase.schema);
        ASSERT_TRUE(parsed);
        const auto compiled =
            rad::JsonSchema::Compile(MakeJsonDocument(parsed.value(), rootUri), options);
        ASSERT_FALSE(compiled);
        EXPECT_EQ(compiled.error().code, testCase.code);
        EXPECT_EQ(compiled.error().schemaPath, testCase.path);
        EXPECT_EQ(compiled.error().schemaUri, rootUri);
    }
    for (const auto otherDraft :
         {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2020_12})
    {
        const auto ignored = rad::JsonSchema::Compile(
            MakeJsonDocument(rad::JsonObject{{"$recursiveRef", false}, {"$recursiveAnchor", 1}}),
            {.dialect = otherDraft});
        ASSERT_TRUE(ignored) << ignored.error().message;
        EXPECT_TRUE(ignored.value().Validate(1));
    }
}

TEST(JsonSchemaReferences, RecursiveRegistryAndCopies)
{
    const auto makeSchema = []
    {
        const std::string rootUri = "https://example.com/strict.json";
        rad::JsonSchemaCompileOptions options{.dialect = rad::JsonSchemaDialect::Draft2019_09};
        options.documents = {
            MakeJsonDocument(
                rad::ParseJson(R"json({"$id": "trees/base.json", "$recursiveAnchor": true,
                 "type": ["object", "integer"],
                 "properties": {"child": {"$recursiveRef": "#"},
                                "static": {"$ref": "#"}}})json")
                    .value(),
                "https://example.com/tree.json"),
        };
        return rad::JsonSchema::Compile(
            MakeJsonDocument(
                rad::JsonObject{{"$recursiveAnchor", true}, {"$ref", "tree.json"}, {"minimum", 2}},
                rootUri),
            options);
    };
    const auto compiled = makeSchema();
    ASSERT_TRUE(compiled) << compiled.error().message;
    auto copied = compiled.value();
    auto moved = std::move(copied);
    for (const std::size_t maxErrors : {0U, 1U, 2U})
    {
        rad::JsonSchemaValidationOptions options;
        options.maxErrors = maxErrors;
        EXPECT_TRUE(moved.Validate(rad::JsonObject{{"child", 2}, {"static", 1}}, options));
        const auto invalid = moved.Validate(rad::JsonObject{{"child", 1}}, options);
        ASSERT_FALSE(invalid);
        ASSERT_EQ(invalid.errors.size(), 1);
        EXPECT_EQ(invalid.errors[0].instancePath, "/child");
        EXPECT_EQ(invalid.errors[0].schemaPath, "/minimum");
        EXPECT_EQ(invalid.errors[0].schemaUri, "https://example.com/strict.json");
        EXPECT_TRUE(moved.Validate(rad::JsonObject{{"child", 2}}, options));
    }
}

TEST(JsonSchemaReferences, RecursiveResourceLimits)
{
    struct Case
    {
        std::string_view schema;
        std::string_view data;
        std::size_t maxEvaluationDepth;
    };
    constexpr Case cases[] = {
        {R"json({"$recursiveAnchor": true, "$recursiveRef": "#"})json", "1", 4},
        {R"json({"$recursiveAnchor": true, "type": "integer",
                 "properties": {"child": {"$recursiveRef": "#"}}})json",
         R"json({"child": {}})json", 3},
    };
    const std::string rootUri = "https://example.com/root.json";
    rad::JsonSchemaCompileOptions compileOptions{.dialect = rad::JsonSchemaDialect::Draft2019_09};
    const auto schema = rad::ParseJson(R"json({"anyOf": [{"$ref": "cycle.json"}, true]})json");
    ASSERT_TRUE(schema);
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto document = rad::ParseJson(testCase.schema);
        const auto data = rad::ParseJson(testCase.data);
        ASSERT_TRUE(document);
        ASSERT_TRUE(data);
        compileOptions.documents = {
            MakeJsonDocument(document.value(), "https://example.com/cycle.json")};
        const auto compiled =
            rad::JsonSchema::Compile(MakeJsonDocument(schema.value(), rootUri), compileOptions);
        ASSERT_TRUE(compiled) << compiled.error().message;
        for (const std::size_t maxErrors : {0U, 1U, 2U})
        {
            rad::JsonSchemaValidationOptions options;
            options.maxErrors = maxErrors;
            options.maxEvaluationDepth = testCase.maxEvaluationDepth;
            const auto result = compiled.value().Validate(data.value(), options);
            if (testCase.maxEvaluationDepth == 3)
            {
                EXPECT_TRUE(result);
                EXPECT_TRUE(result.errors.empty());
                continue;
            }
            ASSERT_FALSE(result);
            ASSERT_EQ(result.errors.size(), 1);
            EXPECT_EQ(result.status, rad::JsonSchemaValidationStatus::EvaluationCycleDetected);
            EXPECT_EQ(result.errors.back().instancePath, "");
            EXPECT_EQ(result.errors.back().schemaPath, "");
            EXPECT_EQ(result.errors.back().schemaUri, "https://example.com/cycle.json");
            EXPECT_EQ(result.errors.back().message,
                      "schema evaluation cycle detected without instance progress");
        }
    }
}

TEST(JsonSchemaReferences, DynamicDiagnostics)
{
    constexpr auto dialect = rad::JsonSchemaDialect::Draft2020_12;
    struct Case
    {
        std::string_view schema;
        std::string_view path;
        rad::JsonSchemaCompileErrorCode code = rad::JsonSchemaCompileErrorCode::InvalidSchema;
    };
    constexpr Case cases[] = {
        {R"json({"$dynamicAnchor": false})json", "/$dynamicAnchor"},
        {R"json({"$dynamicAnchor": ""})json", "/$dynamicAnchor"},
        {R"json({"$dynamicAnchor": "1node"})json", "/$dynamicAnchor"},
        {R"json({"$dynamicAnchor": "node:child"})json", "/$dynamicAnchor"},
        {R"json({"$dynamicAnchor": "node/child"})json", "/$dynamicAnchor"},
        {R"json({"$dynamicRef": 42})json", "/$dynamicRef"},
        {R"json({"$dynamicRef": "#/%ZZ"})json", "/$dynamicRef"},
        {R"json({"$dynamicRef": "#unknown"})json", "/$dynamicRef"},
        {R"json({"$dynamicRef": "#/missing"})json", "/$dynamicRef"},
        {R"json({"properties": {"value": {"$dynamicAnchor": 1}}})json",
         "/properties/value/$dynamicAnchor"},
        {R"json({"$defs": {"one": {"$anchor": "node"},
                          "two": {"$dynamicAnchor": "node"}}})json",
         "/$defs/two/$dynamicAnchor"},
        {R"json({"$defs": {"one": {"$dynamicAnchor": "node"},
                          "two": {"$dynamicAnchor": "node"}}})json",
         "/$defs/two/$dynamicAnchor"},
        {R"json({"$dynamicRef": "external.json#node"})json", "/$dynamicRef",
         rad::JsonSchemaCompileErrorCode::UnsupportedFeature},
    };
    const std::string rootUri = "https://example.com/root.json";
    rad::JsonSchemaCompileOptions options{.dialect = dialect};
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto parsed = rad::ParseJson(testCase.schema);
        ASSERT_TRUE(parsed);
        const auto compiled =
            rad::JsonSchema::Compile(MakeJsonDocument(parsed.value(), rootUri), options);
        ASSERT_FALSE(compiled);
        EXPECT_EQ(compiled.error().code, testCase.code);
        EXPECT_EQ(compiled.error().schemaPath, testCase.path);
        EXPECT_EQ(compiled.error().schemaUri, rootUri);
    }
    EXPECT_TRUE(rad::JsonSchema::Compile(
        MakeJsonDocument(rad::JsonObject{{"$dynamicAnchor", "_node-1.2"}}), {.dialect = dialect}));
    for (const auto otherDraft :
         {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09})
    {
        const auto ignored = rad::JsonSchema::Compile(
            MakeJsonDocument(rad::JsonObject{{"$dynamicRef", false}, {"$dynamicAnchor", 1}}),
            {.dialect = otherDraft});
        ASSERT_TRUE(ignored) << ignored.error().message;
        EXPECT_TRUE(ignored.value().Validate(1));
    }
}

TEST(JsonSchemaReferences, DynamicRegistryAndCopies)
{
    const auto makeSchema = [](std::string_view reference)
    {
        const std::string rootUri = "https://example.com/strict.json";
        rad::JsonSchemaCompileOptions options{.dialect = rad::JsonSchemaDialect::Draft2020_12};
        options.documents = {
            MakeJsonDocument(
                rad::ParseJson(R"json({"$id": "trees/base.json", "$dynamicAnchor": "node",
                 "type": ["object", "integer"],
                 "properties": {"child": {"$dynamicRef": "#node"},
                                "static": {"$ref": "#node"}}})json")
                    .value(),
                "https://example.com/tree.json"),
        };
        return rad::JsonSchema::Compile(
            MakeJsonDocument(
                rad::JsonObject{{"$dynamicAnchor", "node"}, {"$ref", reference}, {"minimum", 2}},
                rootUri),
            options);
    };
    for (const std::string_view reference :
         {"tree.json#node", "trees/base.json#node", "tree.json#%6Eode"})
    {
        SCOPED_TRACE(reference);
        const auto compiled = makeSchema(reference);
        ASSERT_TRUE(compiled) << compiled.error().message;
        auto copied = compiled.value();
        auto moved = std::move(copied);
        for (const std::size_t maxErrors : {0U, 1U, 2U})
        {
            rad::JsonSchemaValidationOptions options;
            options.maxErrors = maxErrors;
            EXPECT_TRUE(moved.Validate(rad::JsonObject{{"child", 2}, {"static", 1}}, options));
            const auto invalid = moved.Validate(rad::JsonObject{{"child", 1}}, options);
            ASSERT_FALSE(invalid);
            ASSERT_EQ(invalid.errors.size(), 1);
            EXPECT_EQ(invalid.errors[0].instancePath, "/child");
            EXPECT_EQ(invalid.errors[0].schemaPath, "/minimum");
            EXPECT_EQ(invalid.errors[0].schemaUri, "https://example.com/strict.json");
            EXPECT_TRUE(moved.Validate(rad::JsonObject{{"child", 2}}, options));
        }
    }
}

TEST(JsonSchemaReferences, DynamicAnchorTargetCompilation)
{
    constexpr auto dialect = rad::JsonSchemaDialect::Draft2020_12;
    const std::string rootUri = "https://example.com/root.json";
    rad::JsonSchemaCompileOptions options{.dialect = dialect};
    const auto document = rad::ParseJson(R"json({"$defs": {
        "entry": {"$dynamicRef": "other.json#node"},
        "hidden": {"$dynamicAnchor": "node", "type": 42}
    }})json");
    ASSERT_TRUE(document);
    options.documents = {
        MakeJsonDocument(document.value(), "https://example.com/child.json"),
        MakeJsonDocument(rad::JsonObject{{"$dynamicAnchor", "node"}, {"type", "string"}},
                         "https://example.com/other.json"),
    };
    const auto root = rad::JsonObject{{"$ref", "child.json#/$defs/entry"}};
    const auto invalid = rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options);
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    EXPECT_EQ(invalid.error().schemaPath, "/$defs/hidden/type");
    EXPECT_EQ(invalid.error().schemaUri, "https://example.com/child.json");
    EXPECT_TRUE(rad::JsonSchema::Compile(MakeJsonDocument(rad::JsonObject{}, rootUri), options));

    auto updatedSchema = options.documents[0]->Root();
    updatedSchema.as_object().at("$defs").as_object().at("hidden").as_object()["type"] = "integer";
    options.documents[0] = MakeJsonDocument(std::move(updatedSchema), options.documents[0]->Uri());
    const auto compiled = rad::JsonSchema::Compile(MakeJsonDocument(root, rootUri), options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    EXPECT_TRUE(compiled.value().Validate(1));
    const auto result = compiled.value().Validate("invalid");
    ASSERT_FALSE(result);
    ASSERT_EQ(result.errors.size(), 1);
    EXPECT_EQ(result.errors[0].schemaPath, "/$defs/hidden/type");
    EXPECT_EQ(result.errors[0].schemaUri, "https://example.com/child.json");
}

TEST(JsonSchemaReferences, DynamicUnevaluatedAnnotations)
{
    struct Case
    {
        std::string_view schema;
        std::string_view valid;
        std::string_view invalid;
        std::string_view instancePath;
        std::string_view schemaPath;
    };
    constexpr Case cases[] = {
        {R"json({"$dynamicAnchor": "node",
                 "allOf": [{"properties": {"known": true}}],
                 "unevaluatedProperties": false})json",
         R"json({"known": 1})json", R"json({"known": 1, "extra": 2})json",
         "/extra", "/$defs/hidden/unevaluatedProperties"},
        {R"json({"$dynamicAnchor": "node",
                 "allOf": [{"prefixItems": [true]}],
                 "unevaluatedItems": false})json",
         "[1]", "[1, 2]", "/1", "/$defs/hidden/unevaluatedItems"},
    };
    const std::string childUri = "https://example.com/child.json";
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto hidden = rad::ParseJson(testCase.schema);
        const auto valid = rad::ParseJson(testCase.valid);
        const auto invalid = rad::ParseJson(testCase.invalid);
        ASSERT_TRUE(hidden);
        ASSERT_TRUE(valid);
        ASSERT_TRUE(invalid);
        rad::JsonSchemaCompileOptions options{.dialect = rad::JsonSchemaDialect::Draft2020_12};
        options.documents = {
            MakeJsonDocument(
                rad::JsonObject{{"$defs", rad::JsonObject{
                    {"entry", rad::JsonObject{{"$dynamicRef", "other.json#node"}}},
                    {"hidden", hidden.value()}}}},
                childUri),
            MakeJsonDocument(rad::JsonObject{{"$dynamicAnchor", "node"}},
                             "https://example.com/other.json"),
        };
        const auto compiled = rad::JsonSchema::Compile(
            MakeJsonDocument(rad::JsonObject{{"$ref", "child.json#/$defs/entry"}},
                             "https://example.com/root.json"),
            options);
        ASSERT_TRUE(compiled) << compiled.error().message;
        auto copied = compiled.value();
        auto moved = std::move(copied);
        for (const std::size_t maxErrors : {1U, 64U})
        {
            SCOPED_TRACE(maxErrors);
            rad::JsonSchemaValidationOptions validation;
            validation.maxErrors = maxErrors;
            EXPECT_TRUE(moved.Validate(valid.value(), validation));
            const auto result = moved.Validate(invalid.value(), validation);
            ASSERT_FALSE(result);
            ASSERT_EQ(result.errors.size(), 1);
            EXPECT_EQ(result.status, maxErrors == 1
                                         ? rad::JsonSchemaValidationStatus::MaxErrorsReached
                                         : rad::JsonSchemaValidationStatus::Completed);
            EXPECT_EQ(result.errors[0].instancePath, testCase.instancePath);
            EXPECT_EQ(result.errors[0].schemaPath, testCase.schemaPath);
            EXPECT_EQ(result.errors[0].schemaUri, childUri);
        }
    }
}

TEST(JsonSchemaReferences, DynamicResourceLimits)
{
    struct Case
    {
        std::string_view schema;
        std::string_view data;
        std::size_t maxEvaluationDepth;
        std::string_view instancePath;
    };
    constexpr Case cases[] = {
        {R"json({"$dynamicAnchor": "node", "$dynamicRef": "#node"})json", "1", 4, ""},
        {R"json({"$dynamicAnchor": "node", "type": "integer",
                 "properties": {"child": {"$dynamicRef": "#node"}}})json",
         R"json({"child": {}})json", 3, "/child"},
    };
    const std::string rootUri = "https://example.com/root.json";
    rad::JsonSchemaCompileOptions compileOptions{.dialect = rad::JsonSchemaDialect::Draft2020_12};
    const auto schema = rad::ParseJson(R"json({"anyOf": [{"$ref": "cycle.json"}, true]})json");
    ASSERT_TRUE(schema);
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto document = rad::ParseJson(testCase.schema);
        const auto data = rad::ParseJson(testCase.data);
        ASSERT_TRUE(document);
        ASSERT_TRUE(data);
        compileOptions.documents = {
            MakeJsonDocument(document.value(), "https://example.com/cycle.json")};
        const auto compiled =
            rad::JsonSchema::Compile(MakeJsonDocument(schema.value(), rootUri), compileOptions);
        ASSERT_TRUE(compiled) << compiled.error().message;
        for (const std::size_t maxErrors : {0U, 1U, 2U})
        {
            rad::JsonSchemaValidationOptions options;
            options.maxErrors = maxErrors;
            options.maxEvaluationDepth = testCase.maxEvaluationDepth;
            const auto result = compiled.value().Validate(data.value(), options);
            if (!testCase.instancePath.empty())
            {
                EXPECT_TRUE(result);
                EXPECT_TRUE(result.errors.empty());
                continue;
            }
            ASSERT_FALSE(result);
            ASSERT_EQ(result.errors.size(), 1);
            EXPECT_EQ(result.status, rad::JsonSchemaValidationStatus::EvaluationCycleDetected);
            EXPECT_EQ(result.errors.back().instancePath, testCase.instancePath);
            EXPECT_EQ(result.errors.back().schemaPath, "");
            EXPECT_EQ(result.errors.back().schemaUri, "https://example.com/cycle.json");
            EXPECT_EQ(result.errors.back().message,
                      "schema evaluation cycle detected without instance progress");
        }
    }
}
