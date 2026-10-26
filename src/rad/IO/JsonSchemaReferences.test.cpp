#include <rad/IO/JsonSchema.h>

#include <gtest/gtest.h>

#include <format>
#include <string>
#include <string_view>
#include <cstddef>
#include <utility>

TEST(IO, JsonSchemaReferenceDiagnosticsAndLimits)
{
    const auto schema = rad::ParseJson(
        R"json({"$defs": {"node": {"type": "object", "properties":
                 {"next": {"$ref": "#/$defs/node"}}}},
                 "$ref": "#/$defs/node"})json");
    ASSERT_TRUE(schema);
    const auto compiled =
        rad::JsonSchema::Compile(schema.value(), {.dialect = rad::JsonSchemaDialect::Draft2020_12});
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
    EXPECT_EQ(limited.errors.back().errorCode,
              rad::JsonSchemaValidationErrorCode::DepthLimitExceeded);
    EXPECT_NE(limited.errors.back().message.find("maximum validation depth"),
              std::string::npos);

    const auto selfReference = rad::ParseJson(R"json({"$ref": "#"})json");
    ASSERT_TRUE(selfReference);
    const auto selfCompiled = rad::JsonSchema::Compile(
        selfReference.value(), {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_TRUE(selfCompiled);
    options.maxErrors = 1;
    const auto selfResult = selfCompiled.value().Validate(1, options);
    EXPECT_FALSE(selfResult);
    ASSERT_EQ(selfResult.errors.size(), 1);
    EXPECT_EQ(selfResult.errors.back().errorCode,
              rad::JsonSchemaValidationErrorCode::DepthLimitExceeded);
    EXPECT_NE(selfResult.errors.back().message.find("maximum validation depth"),
              std::string::npos);
}

TEST(IO, JsonSchemaReferenceResourceLimitsAndCopies)
{
    const auto nestedSource = rad::ParseJson(
        R"json({"$defs": {"inner": {"$id": "sub.json", "$ref": "#"}},
                 "$ref": "#/$defs/inner"})json");
    ASSERT_TRUE(nestedSource);
    const auto nestedCompiled = rad::JsonSchema::Compile(
        nestedSource.value(), {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_TRUE(nestedCompiled) << nestedCompiled.error().message;
    rad::JsonSchemaValidationOptions options;
    options.maxEvaluationDepth = 2;
    const auto recursive = nestedCompiled.value().Validate(1, options);
    ASSERT_FALSE(recursive);
    ASSERT_EQ(recursive.errors.size(), 1);
    EXPECT_EQ(recursive.errors.back().errorCode,
              rad::JsonSchemaValidationErrorCode::DepthLimitExceeded);
    EXPECT_EQ(recursive.errors.back().schemaPath, "/$defs/inner");
    EXPECT_EQ(recursive.errors.back().message, "maximum validation depth exceeded");

    const auto makeSchema = []
    {
        const auto schema = rad::ParseJson(
            R"json({"$defs": {"inner": {"$id": "sub.json", "type": "string"}},
                     "$ref": "sub.json"})json");
        return rad::JsonSchema::Compile(schema.value(),
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

TEST(IO, JsonSchemaResourceIdentifierDiagnostics)
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
            const auto compiled = rad::JsonSchema::Compile(schema.value(), {.dialect = dialect});
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
        const auto compiled = rad::JsonSchema::Compile(deferred.value(), {.dialect = dialect});
        ASSERT_TRUE(compiled) << compiled.error().schemaPath << ": " << compiled.error().message;
        const auto result = compiled.value().Validate(1);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.errors[0].schemaPath, "/storage/$defs/value/type");

        const auto annotations = rad::ParseJson(
            R"json({"const": {"$id": 42, "$anchor": "1invalid"},
                     "examples": [{"$id": "hidden", "$ref": "external"}]})json");
        ASSERT_TRUE(annotations);
        EXPECT_TRUE(rad::JsonSchema::Compile(annotations.value(), {.dialect = dialect}));
    }
    const auto ignored = rad::ParseJson(
        R"json({"$ref": "#/definitions/value", "$id": 42,
                 "definitions": {"value": true, "unused": {"$ref": "external.json"},
                                 "malformed": {"$ref": 42}}})json");
    ASSERT_TRUE(ignored);
    EXPECT_TRUE(
        rad::JsonSchema::Compile(ignored.value(), {.dialect = rad::JsonSchemaDialect::Draft7}));
}

TEST(IO, JsonSchemaReferenceDiscoveryOrder)
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
                const auto compiled = rad::JsonSchema::Compile(value, {.dialect = dialect});
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

TEST(IO, JsonSchemaInvalidReferences)
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
            schema.value(), {.dialect = rad::JsonSchemaDialect::Draft2020_12});
        ASSERT_FALSE(compiled);
        EXPECT_EQ(compiled.error().code, testCase.code);
        EXPECT_EQ(compiled.error().schemaPath, "/$ref");
    }

    const auto nonSchemaTarget = rad::ParseJson(
        R"json({"list": [{"type": "string"}], "$ref": "#/list/0/type"})json");
    ASSERT_TRUE(nonSchemaTarget);
    const auto invalidTarget = rad::JsonSchema::Compile(
        nonSchemaTarget.value(), {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_FALSE(invalidTarget);
    EXPECT_EQ(invalidTarget.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    EXPECT_EQ(invalidTarget.error().schemaPath, "/list/0/type");
}

TEST(IO, JsonSchemaDocumentRegistry)
{
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const auto makeSchema = [&]
        {
            rad::JsonSchemaCompileOptions options{.dialect = dialect};
            options.retrievalUri = "https://example.com/root.json";
            options.documents = {
                {"https://example.com/a.json",
                 rad::ParseJson(R"json({"$id": "canonical/a.json", "type": "object",
                     "properties": {"child": {"$ref": "../b.json#/properties/value"}}})json")
                     .value()},
                {"https://example.com/b.json",
                 rad::ParseJson(R"json({"properties": {"value": {"type": "integer"},
                     "embedded": {"$id": "embedded.json", "type": "string"}}})json")
                     .value()},
            };
            auto root = rad::JsonObject{{"$ref", "canonical/a.json"}};
            if (dialect != rad::JsonSchemaDialect::Draft7)
            {
                root["unevaluatedProperties"] = false;
            }
            return rad::JsonSchema::Compile(root, options);
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

        rad::JsonSchemaCompileOptions options{.dialect = dialect};
        options.retrievalUri = "https://example.com/root.json";
        options.documents = {
            {"https://example.com/container.json", rad::ParseJson(R"json({"properties":
                 {"embedded": {"$id": "embedded.json", "type": "integer"}}})json")
                                                       .value()},
        };
        const auto embedded =
            rad::JsonSchema::Compile(rad::JsonObject{{"$ref", "embedded.json"}}, options);
        ASSERT_TRUE(embedded) << embedded.error().message;
        EXPECT_TRUE(embedded.value().Validate(1));
        const auto failure = embedded.value().Validate("invalid");
        ASSERT_FALSE(failure);
        EXPECT_EQ(failure.errors[0].schemaUri, "https://example.com/container.json");
        EXPECT_EQ(failure.errors[0].schemaPath, "/properties/embedded/type");

        options.documents = {
            {"https://example.com/alias.json", rad::ParseJson(R"json({"$id": "canonical.json",
                                   "properties": {"value": {"$anchor": "value",
                                                           "type": "integer"}}})json")
                                                   .value()},
        };
        if (dialect == rad::JsonSchemaDialect::Draft7)
        {
            auto& value = options.documents[0]
                              .schema.as_object()
                              .at("properties")
                              .as_object()
                              .at("value")
                              .as_object();
            value.erase("$anchor");
            value["$id"] = "#value";
        }
        for (const auto* reference :
             {"alias.json#value", "canonical.json#value", "alias.json#/properties/value"})
        {
            const auto aliased =
                rad::JsonSchema::Compile(rad::JsonObject{{"$ref", reference}}, options);
            ASSERT_TRUE(aliased) << aliased.error().message;
            const auto invalid = aliased.value().Validate("invalid");
            ASSERT_FALSE(invalid);
            EXPECT_EQ(invalid.errors[0].schemaUri, "https://example.com/alias.json");
            EXPECT_EQ(invalid.errors[0].schemaPath, "/properties/value/type");
        }

        options.documents = {
            {"https://example.com/a.json", rad::JsonObject{{"$ref", "b.json"}}},
            {"https://example.com/b.json", rad::JsonObject{{"$ref", "a.json"}}},
        };
        const auto cyclic = rad::JsonSchema::Compile(rad::JsonObject{{"$ref", "a.json"}}, options);
        ASSERT_TRUE(cyclic) << cyclic.error().message;
        rad::JsonSchemaValidationOptions limits;
        limits.maxEvaluationDepth = 2;
        limits.maxErrors = 1;
        const auto exhausted = cyclic.value().Validate(1, limits);
        ASSERT_FALSE(exhausted);
        ASSERT_EQ(exhausted.errors.size(), 1);
        EXPECT_EQ(exhausted.errors.back().errorCode,
                  rad::JsonSchemaValidationErrorCode::DepthLimitExceeded);
        EXPECT_EQ(exhausted.errors.back().schemaPath, "");
        EXPECT_EQ(exhausted.errors.back().schemaUri, "https://example.com/a.json");
        EXPECT_EQ(exhausted.errors.back().message, "maximum validation depth exceeded");

        options.documents = {
            {"https://example.com/a.json",
             rad::ParseJson(R"json({"type": "integer",
                 "properties": {"child": {"allOf": [true]}}})json").value()},
        };
        const auto branches = rad::ParseJson(R"json({"anyOf": [{"$ref": "a.json"}, true]})json");
        ASSERT_TRUE(branches);
        const auto limited = rad::JsonSchema::Compile(branches.value(), options);
        ASSERT_TRUE(limited) << limited.error().message;
        limits.maxEvaluationDepth = 3;
        for (const std::size_t maxErrors : {0U, 1U, 2U})
        {
            limits.maxErrors = maxErrors;
            const auto result =
                limited.value().Validate(rad::JsonObject{{"child", rad::JsonObject{}}}, limits);
            EXPECT_TRUE(result);
            EXPECT_TRUE(result.errors.empty());
        }
        limits.maxEvaluationDepth = 4;
        EXPECT_TRUE(
            limited.value().Validate(rad::JsonObject{{"child", rad::JsonObject{}}}, limits));
    }
}

TEST(IO, JsonSchemaDocumentRegistryDiagnostics)
{
    constexpr auto dialect = rad::JsonSchemaDialect::Draft2020_12;
    rad::JsonSchemaCompileOptions options{.dialect = dialect};
    options.retrievalUri = "https://example.com/root.json";
    const auto root = rad::JsonObject{{"$ref", "child.json"}};
    const auto missing = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    EXPECT_EQ(missing.error().schemaPath, "/$ref");
    EXPECT_EQ(missing.error().schemaUri, options.retrievalUri);

    options.documents = {
        {"https://example.com/child.json", rad::JsonObject{{"type", 42}}},
    };
    const auto invalid = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.error().schemaPath, "/type");
    EXPECT_EQ(invalid.error().schemaUri, "https://example.com/child.json");
    EXPECT_TRUE(rad::JsonSchema::Compile(rad::JsonObject{}, options));

    options.documents[0].schema = rad::JsonObject{
        {"$schema", "http://json-schema.org/draft-07/schema"},
        {"properties", rad::JsonObject{{"child", true}}},
    };
    const auto mixed = rad::JsonSchema::Compile(
        rad::JsonObject{{"$ref", "child.json#/properties/child"}}, options);
    ASSERT_FALSE(mixed);
    EXPECT_EQ(mixed.error().code, rad::JsonSchemaCompileErrorCode::UnsupportedFeature);
    EXPECT_EQ(mixed.error().schemaPath, "/$schema");
    EXPECT_EQ(mixed.error().schemaUri, "https://example.com/child.json");

    options.documents[0].schema = true;
    options.documents.push_back({"https://EXAMPLE.COM/%63hild.json", false});
    const auto duplicate = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(duplicate);
    EXPECT_EQ(duplicate.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    options.documents.pop_back();
    const auto conflict = rad::JsonSchema::Compile(
        rad::JsonObject{{"$id", "https://example.com/child.json"}}, options);
    ASSERT_FALSE(conflict);
    EXPECT_EQ(conflict.error().schemaPath, "/$id");
    EXPECT_EQ(conflict.error().schemaUri, options.retrievalUri);

    auto detected = rad::JsonObject{{"$schema", "https://json-schema.org/draft/2020-12/schema"},
                                    {"$ref", "child.json"}};
    options.dialect = rad::JsonSchemaDialect::Auto;
    EXPECT_TRUE(rad::JsonSchema::Compile(detected, options));
    EXPECT_FALSE(rad::JsonSchema::Compile(detected));
    options.dialect = dialect;
    options.retrievalUri = "relative.json";
    const auto relative = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(relative);
    EXPECT_EQ(relative.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    options.retrievalUri = "https://example.com/root.json#fragment";
    EXPECT_FALSE(rad::JsonSchema::Compile(root, options));

    options.retrievalUri = "https://example.com/root.json";
    options.documents[0].schema = rad::JsonObject{{"$id", 42}};
    const auto ignored = rad::ParseJson(
        R"json({"$ref": "#/definitions/value",
                 "definitions": {"value": true, "unused": {"$ref": "child.json"}}})json");
    ASSERT_TRUE(ignored);
    options.dialect = rad::JsonSchemaDialect::Draft7;
    EXPECT_TRUE(rad::JsonSchema::Compile(ignored.value(), options));
    options.dialect = dialect;
    const auto malformedId = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(malformedId);
    EXPECT_EQ(malformedId.error().schemaPath, "/$id");
    EXPECT_EQ(malformedId.error().schemaUri, "https://example.com/child.json");
}

TEST(IO, JsonSchemaRecursiveReferenceDiagnostics)
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
    rad::JsonSchemaCompileOptions options{.dialect = dialect};
    options.retrievalUri = "https://example.com/root.json";
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto parsed = rad::ParseJson(testCase.schema);
        ASSERT_TRUE(parsed);
        const auto compiled = rad::JsonSchema::Compile(parsed.value(), options);
        ASSERT_FALSE(compiled);
        EXPECT_EQ(compiled.error().code, testCase.code);
        EXPECT_EQ(compiled.error().schemaPath, testCase.path);
        EXPECT_EQ(compiled.error().schemaUri, options.retrievalUri);
    }
    for (const auto otherDraft :
         {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2020_12})
    {
        const auto ignored = rad::JsonSchema::Compile(
            rad::JsonObject{{"$recursiveRef", false}, {"$recursiveAnchor", 1}},
            {.dialect = otherDraft});
        ASSERT_TRUE(ignored) << ignored.error().message;
        EXPECT_TRUE(ignored.value().Validate(1));
    }
}

TEST(IO, JsonSchemaRecursiveReferenceRegistryAndCopies)
{
    const auto makeSchema = []
    {
        rad::JsonSchemaCompileOptions options{.dialect = rad::JsonSchemaDialect::Draft2019_09};
        options.retrievalUri = "https://example.com/strict.json";
        options.documents = {
            {"https://example.com/tree.json",
             rad::ParseJson(R"json({"$id": "trees/base.json", "$recursiveAnchor": true,
                 "type": ["object", "integer"],
                 "properties": {"child": {"$recursiveRef": "#"},
                                "static": {"$ref": "#"}}})json").value()},
        };
        return rad::JsonSchema::Compile(
            rad::JsonObject{{"$recursiveAnchor", true}, {"$ref", "tree.json"}, {"minimum", 2}},
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

TEST(IO, JsonSchemaRecursiveReferenceResourceLimits)
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
    rad::JsonSchemaCompileOptions compileOptions{.dialect = rad::JsonSchemaDialect::Draft2019_09};
    compileOptions.retrievalUri = "https://example.com/root.json";
    const auto schema = rad::ParseJson(R"json({"anyOf": [{"$ref": "cycle.json"}, true]})json");
    ASSERT_TRUE(schema);
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto document = rad::ParseJson(testCase.schema);
        const auto data = rad::ParseJson(testCase.data);
        ASSERT_TRUE(document);
        ASSERT_TRUE(data);
        compileOptions.documents = {{"https://example.com/cycle.json", document.value()}};
        const auto compiled = rad::JsonSchema::Compile(schema.value(), compileOptions);
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
            EXPECT_EQ(result.errors.back().errorCode,
                      rad::JsonSchemaValidationErrorCode::DepthLimitExceeded);
            EXPECT_EQ(result.errors.back().instancePath, "");
            EXPECT_EQ(result.errors.back().schemaPath, "");
            EXPECT_EQ(result.errors.back().schemaUri, "https://example.com/cycle.json");
            EXPECT_EQ(result.errors.back().message, "maximum validation depth exceeded");
        }
    }
}

TEST(IO, JsonSchemaDynamicReferenceDiagnostics)
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
    rad::JsonSchemaCompileOptions options{.dialect = dialect};
    options.retrievalUri = "https://example.com/root.json";
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto parsed = rad::ParseJson(testCase.schema);
        ASSERT_TRUE(parsed);
        const auto compiled = rad::JsonSchema::Compile(parsed.value(), options);
        ASSERT_FALSE(compiled);
        EXPECT_EQ(compiled.error().code, testCase.code);
        EXPECT_EQ(compiled.error().schemaPath, testCase.path);
        EXPECT_EQ(compiled.error().schemaUri, options.retrievalUri);
    }
    EXPECT_TRUE(rad::JsonSchema::Compile(rad::JsonObject{{"$dynamicAnchor", "_node-1.2"}},
                                         {.dialect = dialect}));
    for (const auto otherDraft :
         {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09})
    {
        const auto ignored =
            rad::JsonSchema::Compile(rad::JsonObject{{"$dynamicRef", false}, {"$dynamicAnchor", 1}},
                                     {.dialect = otherDraft});
        ASSERT_TRUE(ignored) << ignored.error().message;
        EXPECT_TRUE(ignored.value().Validate(1));
    }
}

TEST(IO, JsonSchemaDynamicReferenceRegistryAndCopies)
{
    const auto makeSchema = [](std::string_view reference)
    {
        rad::JsonSchemaCompileOptions options{.dialect = rad::JsonSchemaDialect::Draft2020_12};
        options.retrievalUri = "https://example.com/strict.json";
        options.documents = {
            {"https://example.com/tree.json",
             rad::ParseJson(R"json({"$id": "trees/base.json", "$dynamicAnchor": "node",
                 "type": ["object", "integer"],
                 "properties": {"child": {"$dynamicRef": "#node"},
                                "static": {"$ref": "#node"}}})json").value()},
        };
        return rad::JsonSchema::Compile(
            rad::JsonObject{{"$dynamicAnchor", "node"}, {"$ref", reference}, {"minimum", 2}},
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

TEST(IO, JsonSchemaDynamicAnchorTargetCompilation)
{
    constexpr auto dialect = rad::JsonSchemaDialect::Draft2020_12;
    rad::JsonSchemaCompileOptions options{.dialect = dialect};
    options.retrievalUri = "https://example.com/root.json";
    const auto document = rad::ParseJson(R"json({"$defs": {
        "entry": {"$dynamicRef": "other.json#node"},
        "hidden": {"$dynamicAnchor": "node", "type": 42}
    }})json");
    ASSERT_TRUE(document);
    options.documents = {
        {"https://example.com/child.json", document.value()},
        {"https://example.com/other.json",
         rad::JsonObject{{"$dynamicAnchor", "node"}, {"type", "string"}}},
    };
    const auto root = rad::JsonObject{{"$ref", "child.json#/$defs/entry"}};
    const auto invalid = rad::JsonSchema::Compile(root, options);
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    EXPECT_EQ(invalid.error().schemaPath, "/$defs/hidden/type");
    EXPECT_EQ(invalid.error().schemaUri, "https://example.com/child.json");
    EXPECT_TRUE(rad::JsonSchema::Compile(rad::JsonObject{}, options));

    options.documents[0].schema.as_object().at("$defs").as_object()
        .at("hidden").as_object()["type"] = "integer";
    const auto compiled = rad::JsonSchema::Compile(root, options);
    ASSERT_TRUE(compiled) << compiled.error().message;
    EXPECT_TRUE(compiled.value().Validate(1));
    const auto result = compiled.value().Validate("invalid");
    ASSERT_FALSE(result);
    ASSERT_EQ(result.errors.size(), 1);
    EXPECT_EQ(result.errors[0].schemaPath, "/$defs/hidden/type");
    EXPECT_EQ(result.errors[0].schemaUri, "https://example.com/child.json");
}

TEST(IO, JsonSchemaDynamicReferenceResourceLimits)
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
    rad::JsonSchemaCompileOptions compileOptions{.dialect = rad::JsonSchemaDialect::Draft2020_12};
    compileOptions.retrievalUri = "https://example.com/root.json";
    const auto schema = rad::ParseJson(R"json({"anyOf": [{"$ref": "cycle.json"}, true]})json");
    ASSERT_TRUE(schema);
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.schema);
        const auto document = rad::ParseJson(testCase.schema);
        const auto data = rad::ParseJson(testCase.data);
        ASSERT_TRUE(document);
        ASSERT_TRUE(data);
        compileOptions.documents = {{"https://example.com/cycle.json", document.value()}};
        const auto compiled = rad::JsonSchema::Compile(schema.value(), compileOptions);
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
            EXPECT_EQ(result.errors.back().errorCode,
                      rad::JsonSchemaValidationErrorCode::DepthLimitExceeded);
            EXPECT_EQ(result.errors.back().instancePath, testCase.instancePath);
            EXPECT_EQ(result.errors.back().schemaPath, "");
            EXPECT_EQ(result.errors.back().schemaUri, "https://example.com/cycle.json");
            EXPECT_EQ(result.errors.back().message, "maximum validation depth exceeded");
        }
    }
}
