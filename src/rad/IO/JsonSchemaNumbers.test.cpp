#include <rad/IO/JsonSchema.h>

#include "JsonSchema.test.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <format>
#include <string>
#include <string_view>
#include <cmath>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <utility>

using rad::test::FormatValidationErrors;
using rad::test::MakeJsonDocument;

TEST(JsonSchemaNumbers, NonFiniteInstances)
{
    const rad::JsonValue schemas[] = {
        true,
        false,
        rad::JsonObject{},
        rad::JsonObject{{"const", std::int64_t{1}}},
        rad::JsonObject{{"enum", rad::JsonArray{std::numeric_limits<std::uint64_t>::max(), 2.0}}},
        rad::JsonObject{{"anyOf", rad::JsonArray{true, rad::JsonObject{{"minimum", 0}}}}},
    };
    const std::string rootUri = "https://example.com/root.json";
    rad::JsonSchemaCompileOptions compileOptions;
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        compileOptions.dialect = dialect;
        for (const auto& schema : schemas)
        {
            const auto compiled =
                rad::JsonSchema::Compile(MakeJsonDocument(schema, rootUri), compileOptions);
            ASSERT_TRUE(compiled) << compiled.error().message;
            for (const double value :
                 {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()})
            {
                const auto result = compiled.value().Validate(value);
                EXPECT_FALSE(result);
                ASSERT_EQ(result.errors.size(), 1);
                EXPECT_EQ(result.status, rad::JsonSchemaValidationStatus::Completed);
                EXPECT_EQ(result.errors[0].message, "number must be finite");
                EXPECT_EQ(result.errors[0].instancePath, "");
                EXPECT_EQ(result.errors[0].schemaPath, "");
                EXPECT_EQ(result.errors[0].schemaUri, rootUri);
            }
        }
        const auto permissive =
            rad::JsonSchema::Compile(MakeJsonDocument(true, rootUri), compileOptions);
        ASSERT_TRUE(permissive);
        const rad::JsonValue nested =
            rad::JsonObject{{"x/~", rad::JsonArray{1, std::numeric_limits<double>::quiet_NaN()}},
                            {"other", std::numeric_limits<double>::infinity()}};
        for (const std::size_t maxErrors : {0U, 1U, 2U, 64U})
        {
            rad::JsonSchemaValidationOptions options;
            options.maxErrors = maxErrors;
            options.maxEvaluationDepth = 0;
            const auto result = permissive.value().Validate(nested, options);
            EXPECT_FALSE(result);
            ASSERT_EQ(result.errors.size(), maxErrors <= 1 ? 1 : 2);
            EXPECT_EQ(result.status, maxErrors <= 2
                                         ? rad::JsonSchemaValidationStatus::MaxErrorsReached
                                         : rad::JsonSchemaValidationStatus::Completed);
            EXPECT_EQ(result.errors[0].instancePath, "/x~1~0/1");
            EXPECT_EQ(result.errors[0].schemaPath, "");
            if (result.errors.size() == 2)
            {
                EXPECT_EQ(result.errors[1].instancePath, "/other");
            }
        }
        rad::JsonValue deep = 1;
        std::string deepPath;
        for (std::size_t depth = 0; depth < 256; ++depth)
        {
            deep = rad::JsonArray{std::move(deep)};
            deepPath += "/0";
        }
        rad::JsonSchemaValidationOptions options;
        options.maxEvaluationDepth = 0;
        EXPECT_TRUE(permissive.value().Validate(deep, options));
        auto* leaf = &deep;
        for (std::size_t depth = 0; depth < 256; ++depth)
        {
            leaf = &leaf->as_array()[0];
        }
        *leaf = std::numeric_limits<double>::quiet_NaN();
        const auto deepInvalid = permissive.value().Validate(deep, options);
        EXPECT_EQ(deepInvalid.status, rad::JsonSchemaValidationStatus::Completed);
        ASSERT_EQ(deepInvalid.errors.size(), 1);
        EXPECT_EQ(deepInvalid.errors[0].instancePath, deepPath);
        EXPECT_EQ(deepInvalid.errors[0].message, "number must be finite");
    }
}

TEST(JsonSchemaNumbers, NonFiniteLiterals)
{
    const std::string rootUri = "https://example.com/root.json";
    rad::JsonSchemaCompileOptions options;
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        options.dialect = dialect;
        for (const double value :
             {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity()})
        {
            struct Case
            {
                rad::JsonValue schema;
                std::string_view path;
            };
            const Case cases[] = {
                {rad::JsonObject{{"const", value}}, "/const"},
                {rad::JsonObject{{"enum", rad::JsonArray{1, value, value}}}, "/enum/1"},
                {rad::JsonObject{{"const", rad::JsonObject{{"/~", rad::JsonArray{value}}}}},
                 "/const/~1~0/0"},
                {rad::JsonObject{
                     {"enum", rad::JsonArray{1, rad::JsonObject{{"/~", rad::JsonArray{value}}}}}},
                 "/enum/1/~1~0/0"},
            };
            for (const auto& testCase : cases)
            {
                const auto compiled =
                    rad::JsonSchema::Compile(MakeJsonDocument(testCase.schema, rootUri), options);
                ASSERT_FALSE(compiled);
                EXPECT_EQ(compiled.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
                EXPECT_EQ(compiled.error().schemaPath, testCase.path);
                EXPECT_EQ(compiled.error().schemaUri, rootUri);
            }
        }
        const auto nan = std::numeric_limits<double>::quiet_NaN();
        for (const rad::JsonArray names :
             {rad::JsonArray{std::int64_t{1}, nan}, rad::JsonArray{std::uint64_t{1}, nan},
              rad::JsonArray{nan, std::int64_t{1}}, rad::JsonArray{nan, std::uint64_t{1}},
              rad::JsonArray{nan, nan}})
        {
            const auto invalidNames = rad::JsonSchema::Compile(
                MakeJsonDocument(rad::JsonObject{{"required", names}}, rootUri), options);
            ASSERT_FALSE(invalidNames);
            EXPECT_EQ(invalidNames.error().schemaPath, "/required/0");
        }
        options.documents = {
            MakeJsonDocument(rad::JsonObject{{"const", std::numeric_limits<double>::quiet_NaN()}},
                             "https://example.com/unused.json"),
        };
        EXPECT_TRUE(rad::JsonSchema::Compile(MakeJsonDocument(true, rootUri), options));
        const auto referenced = rad::JsonSchema::Compile(
            MakeJsonDocument(rad::JsonObject{{"$ref", "unused.json"}}, rootUri), options);
        ASSERT_FALSE(referenced);
        EXPECT_EQ(referenced.error().schemaPath, "/const");
        EXPECT_EQ(referenced.error().schemaUri, options.documents[0]->Uri());
    }
    options.documents = {
        MakeJsonDocument(
            rad::JsonObject{
                {"$schema", "https://json-schema.org/draft/2020-12/schema"},
                {"$vocabulary",
                 rad::JsonObject{{"https://json-schema.org/draft/2020-12/vocab/core", true}}}},
            "https://example.com/meta"),
    };
    options.dialect = rad::JsonSchemaDialect::Draft2020_12;
    const auto disabled = rad::JsonSchema::Compile(
        MakeJsonDocument(
            rad::JsonObject{{"$schema", "https://example.com/meta"},
                            {"const", std::numeric_limits<double>::quiet_NaN()},
                            {"enum", rad::JsonArray{std::numeric_limits<double>::infinity()}}},
            rootUri),
        options);
    ASSERT_TRUE(disabled) << disabled.error().message;
    EXPECT_TRUE(disabled.value().Validate(1));
    EXPECT_FALSE(disabled.value().Validate(std::numeric_limits<double>::quiet_NaN()));
}

TEST(JsonSchemaNumbers, ComparisonBoundaries)
{
    struct Case
    {
        rad::JsonValue value;
        rad::JsonValue limit;
        int ordering;
    };
    const Case cases[] = {
        {std::int64_t{1}, std::uint64_t{1}, 0},
        {std::int64_t{-1}, std::numeric_limits<std::uint64_t>::max(), -1},
        {std::numeric_limits<std::uint64_t>::max(), std::int64_t{-1}, 1},
        {std::numeric_limits<std::int64_t>::min(), -9223372036854775808.0, 0},
        {std::numeric_limits<std::int64_t>::max(), 9223372036854775808.0, -1},
        {9223372036854775808.0, std::numeric_limits<std::int64_t>::max(), 1},
        {std::numeric_limits<std::uint64_t>::max(), 18446744073709551616.0, -1},
        {18446744073709551616.0, std::numeric_limits<std::uint64_t>::max(), 1},
        {std::nextafter(18446744073709551616.0, 0.0), std::uint64_t{18446744073709549568ULL}, 0},
        {-1.5, std::int64_t{-1}, -1},
        {1.5, std::uint64_t{1}, 1},
        {-0.0, std::uint64_t{0}, 0},
    };
    for (const auto& testCase : cases)
    {
        for (const auto keyword : {"const", "minimum", "maximum"})
        {
            const auto compiled = rad::JsonSchema::Compile(
                MakeJsonDocument(rad::JsonObject{{keyword, testCase.limit}}),
                {.dialect = rad::JsonSchemaDialect::Draft2020_12});
            ASSERT_TRUE(compiled);
            const bool expected = std::string_view(keyword) == "const"     ? testCase.ordering == 0
                                  : std::string_view(keyword) == "minimum" ? testCase.ordering >= 0
                                                                           : testCase.ordering <= 0;
            EXPECT_EQ(static_cast<bool>(compiled.value().Validate(testCase.value)), expected);
        }
    }
}

TEST(JsonSchemaNumbers, SizeConstraints)
{
    struct Constraint
    {
        const char* keyword;
        rad::JsonValue instance;
        bool minimum;
        bool contains = false;
    };
    const Constraint constraints[] = {
        {"minLength", "ab", true},
        {"maxLength", "ab", false},
        {"minItems", rad::JsonArray{1, 2}, true},
        {"maxItems", rad::JsonArray{1, 2}, false},
        {"minProperties", rad::JsonObject{{"a", 1}, {"b", 2}}, true},
        {"maxProperties", rad::JsonObject{{"a", 1}, {"b", 2}}, false},
        {"minContains", rad::JsonArray{1, "other", 2}, true, true},
        {"maxContains", rad::JsonArray{1, "other", 2}, false, true},
    };
    struct Limit
    {
        rad::JsonValue value;
        bool minimumValid;
        bool maximumValid;
    };
    const Limit limits[] = {
        {0, true, false},
        {-0.0, true, false},
        {std::uint64_t{1}, true, false},
        {2, true, true},
        {std::uint64_t{2}, true, true},
        {2.0, true, true},
        {3.0, false, true},
        {std::uint64_t{4294967296ULL}, false, true},
        {9007199254740992.0, false, true},
        {std::uint64_t{9007199254740993ULL}, false, true},
        {std::numeric_limits<std::uint64_t>::max(), false, true},
        {std::nextafter(18446744073709551616.0, 0.0), false, true},
        {18446744073709551616.0, false, true},
        {1e100, false, true},
        {std::numeric_limits<double>::max(), false, true},
    };
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const std::string rootUri = "https://example.com/sizes.json";
        rad::JsonSchemaCompileOptions options{.dialect = dialect};
        for (const auto& constraint : constraints)
        {
            SCOPED_TRACE(constraint.keyword);
            for (const auto& limit : limits)
            {
                SCOPED_TRACE(rad::PrettyJson(limit.value));
                rad::JsonObject schema{{constraint.keyword, limit.value}};
                if (constraint.contains)
                {
                    schema.emplace("contains", rad::JsonObject{{"type", "integer"}});
                }
                const auto compiled =
                    rad::JsonSchema::Compile(MakeJsonDocument(schema, rootUri), options);
                ASSERT_TRUE(compiled) << compiled.error().message;
                const bool expected =
                    (constraint.contains && dialect == rad::JsonSchemaDialect::Draft7) ||
                    (constraint.minimum ? limit.minimumValid : limit.maximumValid);
                for (const std::size_t maxErrors : {1U, 64U})
                {
                    const auto result =
                        compiled.value().Validate(constraint.instance, {.maxErrors = maxErrors});
                    EXPECT_EQ(static_cast<bool>(result), expected) << FormatValidationErrors(result);
                    EXPECT_EQ(result.status,
                              !expected && maxErrors == 1
                                  ? rad::JsonSchemaValidationStatus::MaxErrorsReached
                                  : rad::JsonSchemaValidationStatus::Completed);
                    ASSERT_EQ(result.errors.size(), expected ? 0U : 1U);
                    if (!expected)
                    {
                        EXPECT_EQ(result.errors[0].instancePath, "");
                        EXPECT_EQ(result.errors[0].schemaUri, rootUri);
                        EXPECT_EQ(result.errors[0].schemaPath,
                                  "/" + std::string(constraint.keyword));
                    }
                }
            }
            if (constraint.contains)
            {
                const auto inactive = rad::JsonSchema::Compile(
                    MakeJsonDocument(rad::JsonObject{{constraint.keyword, 1e100}}, rootUri),
                    options);
                ASSERT_TRUE(inactive);
                EXPECT_TRUE(inactive.value().Validate(constraint.instance));
            }
        }
        if (dialect != rad::JsonSchemaDialect::Draft7)
        {
            rad::JsonObject schema{{"contains", false}, {"maxContains", 1e100}};
            const auto defaultMinimum =
                rad::JsonSchema::Compile(MakeJsonDocument(schema, rootUri), options);
            ASSERT_TRUE(defaultMinimum);
            const auto rejected = defaultMinimum.value().Validate(rad::JsonArray{});
            ASSERT_EQ(rejected.errors.size(), 1);
            EXPECT_EQ(rejected.errors[0].schemaPath, "/contains");
            schema.emplace("minContains", -0.0);
            const auto zeroMinimum =
                rad::JsonSchema::Compile(MakeJsonDocument(schema, rootUri), options);
            ASSERT_TRUE(zeroMinimum);
            EXPECT_TRUE(zeroMinimum.value().Validate(rad::JsonArray{}));
        }
    }
}

TEST(JsonSchemaNumbers, InvalidSizeConstraints)
{
    const rad::JsonValue invalid[] = {
        -1, -0.5, 0.5, std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(), nullptr, true, "2", rad::JsonArray{},
        rad::JsonObject{},
    };
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        const std::string rootUri = "https://example.com/invalid-sizes.json";
        rad::JsonSchemaCompileOptions options{.dialect = dialect};
        for (const std::string_view keyword :
             {"minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties",
              "minContains", "maxContains"})
        {
            SCOPED_TRACE(keyword);
            for (const auto& value : invalid)
            {
                rad::JsonObject schema{{keyword, value}};
                if (keyword.ends_with("Contains"))
                {
                    schema.emplace("contains", true);
                }
                const auto compiled =
                    rad::JsonSchema::Compile(MakeJsonDocument(schema, rootUri), options);
                if (dialect == rad::JsonSchemaDialect::Draft7 && keyword.ends_with("Contains"))
                {
                    ASSERT_TRUE(compiled);
                    EXPECT_TRUE(compiled.value().Validate(rad::JsonArray{1}));
                    EXPECT_FALSE(compiled.value().Validate(rad::JsonArray{}));
                    continue;
                }
                ASSERT_FALSE(compiled);
                EXPECT_EQ(compiled.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
                EXPECT_EQ(compiled.error().schemaUri, rootUri);
                EXPECT_EQ(compiled.error().schemaPath, "/" + std::string(keyword));
                EXPECT_EQ(compiled.error().message,
                          std::string(keyword) + " must be a non-negative integer");
            }
        }
    }
}

TEST(JsonSchemaNumbers, MultipleOfDecimalSemantics)
{
    struct Case
    {
        rad::JsonValue value;
        rad::JsonValue divisor;
        bool valid;
    };
    const Case cases[] = {
        {0.3, 0.1, true},
        {0.1 + 0.2, 0.1, false},
        {std::nextafter(0.3, 0.0), 0.1, false},
        {std::nextafter(0.3, 1.0), 0.1, false},
        {0.3, std::nextafter(0.1, 0.0), false},
        {0.3, std::nextafter(0.1, 1.0), false},
        {-0.0, std::numeric_limits<double>::denorm_min(), true},
        {std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::denorm_min(),
         true},
        {std::numeric_limits<double>::denorm_min(), 1e-323, false},
        {1e-323, std::numeric_limits<double>::denorm_min(), true},
        {std::numeric_limits<double>::max(), 0.5, true},
        {std::numeric_limits<double>::max(), 3, false},
        {std::numeric_limits<std::uint64_t>::max(), 0.5, true},
        {std::numeric_limits<std::uint64_t>::max(), 10.0, false},
        {std::numeric_limits<std::int64_t>::min(), 0.5, true},
        {std::numeric_limits<std::int64_t>::min(), 3, false},
        {0.125, 0.25, false},
        {0.25, 0.125, true},
        {1e-300, 1e300, false},
        {1e300, 1e-300, true},
    };
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        for (const auto& testCase : cases)
        {
            SCOPED_TRACE(rad::PrettyJson(testCase.value));
            SCOPED_TRACE(rad::PrettyJson(testCase.divisor));
            const auto compiled = rad::JsonSchema::Compile(
                MakeJsonDocument(rad::JsonObject{{"multipleOf", testCase.divisor}}),
                {.dialect = dialect});
            ASSERT_TRUE(compiled) << compiled.error().message;
            rad::JsonSchemaValidationOptions options;
            options.maxErrors = 1;
            const auto result = compiled.value().Validate(testCase.value, options);
            EXPECT_EQ(static_cast<bool>(result), testCase.valid) << FormatValidationErrors(result);
            if (!testCase.valid)
            {
                ASSERT_EQ(result.errors.size(), 1);
                EXPECT_EQ(result.errors[0].schemaPath, "/multipleOf");
            }
        }
        for (const double divisor : {0.0, -0.1, std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN()})
        {
            const auto compiled = rad::JsonSchema::Compile(
                MakeJsonDocument(rad::JsonObject{{"multipleOf", divisor}}), {.dialect = dialect});
            ASSERT_FALSE(compiled);
            EXPECT_EQ(compiled.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
            EXPECT_EQ(compiled.error().schemaPath, "/multipleOf");
        }
    }
}

TEST(JsonSchemaNumbers, MultipleOfNestedDiagnostics)
{
    struct Case
    {
        rad::JsonValue divisor;
        rad::JsonValue valid;
        rad::JsonValue invalid;
    };
    const Case cases[] = {{3, 6, 7}, {0.1, 0.3, 0.1 + 0.2}};
    const std::string uri = "https://example.com/numbers.json";
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        SCOPED_TRACE(static_cast<int>(dialect));
        for (const auto& testCase : cases)
        {
            SCOPED_TRACE(rad::PrettyJson(testCase.divisor));
            const auto compiled = rad::JsonSchema::Compile(
                MakeJsonDocument(
                    rad::JsonObject{{"properties", rad::JsonObject{
                        {"accounting/records~", rad::JsonObject{
                            {"items", rad::JsonObject{{"multipleOf", testCase.divisor}}}}}}}},
                    uri),
                {.dialect = dialect});
            ASSERT_TRUE(compiled) << compiled.error().message;
            const rad::JsonValue valid =
                rad::JsonObject{{"accounting/records~", rad::JsonArray{testCase.valid}}};
            const rad::JsonValue invalid =
                rad::JsonObject{{"accounting/records~", rad::JsonArray{testCase.invalid}}};
            for (const std::size_t maxErrors : {0U, 1U, 64U})
            {
                rad::JsonSchemaValidationOptions options;
                options.maxErrors = maxErrors;
                EXPECT_TRUE(compiled.value().Validate(valid, options));
                const auto result = compiled.value().Validate(invalid, options);
                ASSERT_FALSE(result);
                ASSERT_EQ(result.errors.size(), 1);
                EXPECT_EQ(result.status, maxErrors <= 1
                                             ? rad::JsonSchemaValidationStatus::MaxErrorsReached
                                             : rad::JsonSchemaValidationStatus::Completed);
                EXPECT_EQ(result.errors[0].instancePath, "/accounting~1records~0/0");
                EXPECT_EQ(result.errors[0].schemaUri, uri);
                EXPECT_EQ(result.errors[0].schemaPath,
                          "/properties/accounting~1records~0/items/multipleOf");
                EXPECT_EQ(result.errors[0].message,
                          "number is not a multiple of the required value");
            }
        }
    }
}

TEST(JsonSchemaNumbers, MultipleOfDecimalGrid)
{
    const auto powerOfTen = [](int exponent)
    {
        std::int64_t power = 1;
        for (int index = 0; index < exponent; ++index)
        {
            power *= 10;
        }
        return power;
    };
    for (int divisorCoefficient = 1; divisorCoefficient <= 25; ++divisorCoefficient)
    {
        for (int divisorExponent = -3; divisorExponent <= 3; ++divisorExponent)
        {
            const auto divisor =
                rad::ParseJson(std::format("{}e{}", divisorCoefficient, divisorExponent));
            ASSERT_TRUE(divisor);
            const auto compiled = rad::JsonSchema::Compile(
                MakeJsonDocument(rad::JsonObject{{"multipleOf", divisor.value()}}),
                {.dialect = rad::JsonSchemaDialect::Draft2020_12});
            ASSERT_TRUE(compiled) << compiled.error().message;
            for (int coefficient = -50; coefficient <= 50; ++coefficient)
            {
                for (int exponent = -3; exponent <= 3; ++exponent)
                {
                    const auto text = std::format("{}e{}", coefficient, exponent);
                    const auto instance = rad::ParseJson(text);
                    ASSERT_TRUE(instance);
                    const int difference = exponent - divisorExponent;
                    const auto numerator = coefficient * powerOfTen(std::max(difference, 0));
                    const auto denominator =
                        divisorCoefficient * powerOfTen(std::max(-difference, 0));
                    const bool expected = numerator % denominator == 0;
                    const auto result = compiled.value().Validate(instance.value());
                    EXPECT_EQ(static_cast<bool>(result), expected)
                        << text << " / " << rad::PrettyJson(divisor.value()) << "\n"
                        << FormatValidationErrors(result);
                }
            }
        }
    }
}
