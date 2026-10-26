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

TEST(IO, JsonSchemaNonFiniteInstances)
{
    const rad::JsonValue schemas[] = {
        true,
        false,
        rad::JsonObject{},
        rad::JsonObject{{"const", std::int64_t{1}}},
        rad::JsonObject{{"enum", rad::JsonArray{std::numeric_limits<std::uint64_t>::max(), 2.0}}},
        rad::JsonObject{{"anyOf", rad::JsonArray{true, rad::JsonObject{{"minimum", 0}}}}},
    };
    rad::JsonSchemaCompileOptions compileOptions;
    compileOptions.retrievalUri = "https://example.com/root.json";
    for (const auto dialect : {rad::JsonSchemaDialect::Draft7, rad::JsonSchemaDialect::Draft2019_09,
                               rad::JsonSchemaDialect::Draft2020_12})
    {
        compileOptions.dialect = dialect;
        for (const auto& schema : schemas)
        {
            const auto compiled = rad::JsonSchema::Compile(schema, compileOptions);
            ASSERT_TRUE(compiled) << compiled.error().message;
            for (const double value :
                 {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()})
            {
                const auto result = compiled.value().Validate(value);
                EXPECT_FALSE(result);
                ASSERT_EQ(result.errors.size(), 1);
                EXPECT_EQ(result.errors[0].errorCode,
                          rad::JsonSchemaValidationErrorCode::SchemaViolation);
                EXPECT_EQ(result.errors[0].message, "number must be finite");
                EXPECT_EQ(result.errors[0].instancePath, "");
                EXPECT_EQ(result.errors[0].schemaPath, "");
                EXPECT_EQ(result.errors[0].schemaUri, compileOptions.retrievalUri);
            }
        }
        const auto permissive = rad::JsonSchema::Compile(true, compileOptions);
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
            ASSERT_EQ(result.errors.size(), maxErrors <= 1 ? 1 : 2);
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
        ASSERT_EQ(deepInvalid.errors.size(), 1);
        EXPECT_EQ(deepInvalid.errors[0].instancePath, deepPath);
        EXPECT_EQ(deepInvalid.errors[0].message, "number must be finite");
    }
}

TEST(IO, JsonSchemaNonFiniteLiterals)
{
    rad::JsonSchemaCompileOptions options;
    options.retrievalUri = "https://example.com/root.json";
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
                const auto compiled = rad::JsonSchema::Compile(testCase.schema, options);
                ASSERT_FALSE(compiled);
                EXPECT_EQ(compiled.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
                EXPECT_EQ(compiled.error().schemaPath, testCase.path);
                EXPECT_EQ(compiled.error().schemaUri, options.retrievalUri);
            }
        }
        const auto nan = std::numeric_limits<double>::quiet_NaN();
        for (const rad::JsonArray names :
             {rad::JsonArray{std::int64_t{1}, nan}, rad::JsonArray{std::uint64_t{1}, nan},
              rad::JsonArray{nan, std::int64_t{1}}, rad::JsonArray{nan, std::uint64_t{1}},
              rad::JsonArray{nan, nan}})
        {
            const auto invalidNames =
                rad::JsonSchema::Compile(rad::JsonObject{{"required", names}}, options);
            ASSERT_FALSE(invalidNames);
            EXPECT_EQ(invalidNames.error().schemaPath, "/required/0");
        }
        options.documents = {
            {"https://example.com/unused.json",
             rad::JsonObject{{"const", std::numeric_limits<double>::quiet_NaN()}}},
        };
        EXPECT_TRUE(rad::JsonSchema::Compile(true, options));
        const auto referenced =
            rad::JsonSchema::Compile(rad::JsonObject{{"$ref", "unused.json"}}, options);
        ASSERT_FALSE(referenced);
        EXPECT_EQ(referenced.error().schemaPath, "/const");
        EXPECT_EQ(referenced.error().schemaUri, options.documents[0].uri);
    }
    options.documents = {
        {"https://example.com/meta",
         rad::JsonObject{
             {"$schema", "https://json-schema.org/draft/2020-12/schema"},
             {"$vocabulary",
              rad::JsonObject{{"https://json-schema.org/draft/2020-12/vocab/core", true}}}}},
    };
    options.dialect = rad::JsonSchemaDialect::Auto;
    const auto disabled = rad::JsonSchema::Compile(
        rad::JsonObject{{"$schema", "https://example.com/meta"},
                        {"const", std::numeric_limits<double>::quiet_NaN()},
                        {"enum", rad::JsonArray{std::numeric_limits<double>::infinity()}}},
        options);
    ASSERT_TRUE(disabled) << disabled.error().message;
    EXPECT_TRUE(disabled.value().Validate(1));
    EXPECT_FALSE(disabled.value().Validate(std::numeric_limits<double>::quiet_NaN()));
}

TEST(IO, JsonSchemaNumericComparisonBoundaries)
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
            const auto compiled =
                rad::JsonSchema::Compile(rad::JsonObject{{keyword, testCase.limit}},
                                         {.dialect = rad::JsonSchemaDialect::Draft2020_12});
            ASSERT_TRUE(compiled);
            const bool expected = std::string_view(keyword) == "const"     ? testCase.ordering == 0
                                  : std::string_view(keyword) == "minimum" ? testCase.ordering >= 0
                                                                           : testCase.ordering <= 0;
            EXPECT_EQ(static_cast<bool>(compiled.value().Validate(testCase.value)), expected);
        }
    }
}

TEST(IO, JsonSchemaMultipleOfDecimalSemantics)
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
                rad::JsonObject{{"multipleOf", testCase.divisor}}, {.dialect = dialect});
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
            const auto compiled = rad::JsonSchema::Compile(rad::JsonObject{{"multipleOf", divisor}},
                                                           {.dialect = dialect});
            ASSERT_FALSE(compiled);
            EXPECT_EQ(compiled.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
            EXPECT_EQ(compiled.error().schemaPath, "/multipleOf");
        }
    }
}

TEST(IO, JsonSchemaMultipleOfDecimalGrid)
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
            const auto compiled =
                rad::JsonSchema::Compile(rad::JsonObject{{"multipleOf", divisor.value()}},
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
