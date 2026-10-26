#include <rad/IO/JsonSchema.h>
#include <rad/Core/Unicode.h>

#include "JsonSchema.test.h"

#include <gtest/gtest.h>

#include <format>
#include <string>
#include <string_view>
#include <vector>
#include <future>
#include <cstddef>
#include <cstdint>
#include <utility>

using rad::test::FormatValidationErrors;

#if !defined(RAD_JSON_SCHEMA_USE_STD_REGEX) || !RAD_JSON_SCHEMA_USE_STD_REGEX
TEST(IO, JsonSchemaRegexWhitespaceCompatibility)
{
    struct Pattern
    {
        std::string_view text;
        bool complement;
        bool includeUnderscore;
    };
    constexpr Pattern patterns[] = {
        {R"(^\s$)", false, false},
        {R"(^\S$)", true, false},
        {R"(^[\s]$)", false, false},
        {R"(^[^\S]$)", false, false},
        {R"(^[\S]$)", true, false},
        {R"(^[^\s]$)", true, false},
        {R"(^[\s_]$)", false, true},
        {R"(^[_\S]$)", true, true},
        {R"(^[\s\S]$)", true, true},
    };
    constexpr char32_t boundaries[] = {
        0x0009, 0x000d, 0x0020, 0x00a0, 0x1680, 0x2000, 0x200a,
        0x2028, 0x2029, 0x202f, 0x205f, 0x3000, 0xfeff,
    };
    std::vector<char32_t> points = {0, 0x0085, 0x180e, U'_', U'a',
                                   0xd7ff, 0xe000, 0x10000, 0x10ffff};
    for (const char32_t boundary : boundaries)
    {
        points.push_back(boundary - 1);
        points.push_back(boundary);
        points.push_back(boundary + 1);
    }
    const auto isWhitespace = [](char32_t point) {
        return (point >= 0x0009 && point <= 0x000d) || point == 0x0020 ||
               point == 0x00a0 || point == 0x1680 ||
               (point >= 0x2000 && point <= 0x200a) ||
               point == 0x2028 || point == 0x2029 || point == 0x202f ||
               point == 0x205f || point == 0x3000 || point == 0xfeff;
    };
    for (const auto& pattern : patterns)
    {
        SCOPED_TRACE(pattern.text);
        const auto compiled =
            rad::JsonSchema::Compile(rad::JsonObject{{"pattern", pattern.text}},
                                     {.dialect = rad::JsonSchemaDialect::Draft2020_12});
        ASSERT_TRUE(compiled) << compiled.error().message;
        for (const char32_t point : points)
        {
            SCOPED_TRACE(static_cast<std::uint32_t>(point));
            const bool expected = pattern.text == R"(^[\s\S]$)" ||
                                  (isWhitespace(point) != pattern.complement) ||
                                  (pattern.includeUnderscore && point == U'_');
            const auto result = compiled.value().Validate(
                rad::JsonValue(rad::Utf32ToUtf8(std::u32string_view(&point, 1))));
            EXPECT_EQ(static_cast<bool>(result), expected) << FormatValidationErrors(result);
        }
    }

    struct Literal
    {
        std::string_view pattern;
        std::string_view value;
    };
    constexpr Literal literals[] = {
        {R"(^\\s$)", R"(\s)"},
        {R"(^\Q\s\S\E$)", R"(\s\S)"},
        {R"(^[\s\-]$)", "-"},
        {R"(^[-\s]$)", "-"},
        {R"(^[^\-\S]$)", " "},
        {R"(^\[\s\]$)", "[ ]"},
        {R"(^[[:digit:]\s]$)", " "},
        {R"(^(?# [\s)\s$)", " "},
        {R"(^\c[\s$)", "\x1b "},
    };
    for (const auto& literal : literals)
    {
        SCOPED_TRACE(literal.pattern);
        const auto compiled =
            rad::JsonSchema::Compile(rad::JsonObject{{"pattern", literal.pattern}},
                                     {.dialect = rad::JsonSchemaDialect::Draft2020_12});
        ASSERT_TRUE(compiled) << compiled.error().message;
        EXPECT_TRUE(compiled.value().Validate(rad::JsonValue(literal.value)));
    }
    for (const auto* pattern : {R"([a-\s])", R"([\S-a])", R"([\s-\S])"})
    {
        SCOPED_TRACE(pattern);
        const auto compiled =
            rad::JsonSchema::Compile(rad::JsonObject{{"pattern", pattern}},
                                     {.dialect = rad::JsonSchemaDialect::Draft2020_12});
        ASSERT_FALSE(compiled);
        EXPECT_EQ(compiled.error().code, rad::JsonSchemaCompileErrorCode::InvalidSchema);
    }
}
#endif

TEST(IO, JsonSchemaRegexCacheLifetimeAndConcurrency)
{
    const auto makeSchema = []
    {
        rad::JsonSchemaCompileOptions options{.dialect = rad::JsonSchemaDialect::Draft2020_12};
        options.retrievalUri = "https://example.com/root.json";
        options.documents = {
            {"https://example.com/remote.json", rad::JsonObject{{"pattern", "^remote$"}}},
        };
        return rad::JsonSchema::Compile(
            rad::JsonObject{
                {"properties",
                 rad::JsonObject{{"local", rad::JsonObject{{"pattern", "^local$"}}},
                                 {"remote", rad::JsonObject{{"$ref", "remote.json"}}}}},
                {"patternProperties",
                 rad::JsonObject{{"^x/~", rad::JsonObject{{"pattern", "^extra$"}}}}},
            },
            options);
    };
    const auto compiled = makeSchema();
    ASSERT_TRUE(compiled) << compiled.error().message;
    auto copied = compiled.value();
    auto moved = std::move(copied);
    const rad::JsonValue valid =
        rad::JsonObject{{"local", "local"}, {"remote", "remote"}, {"x/~tag", "extra"}};
    const rad::JsonValue invalid =
        rad::JsonObject{{"local", "remote"}, {"remote", "local"}, {"x/~tag", "bad"}};
    const auto result = moved.Validate(invalid);
    ASSERT_EQ(result.errors.size(), 3);
    EXPECT_EQ(result.errors[0].schemaPath, "/properties/local/pattern");
    EXPECT_EQ(result.errors[0].schemaUri, "https://example.com/root.json");
    EXPECT_EQ(result.errors[1].schemaPath, "/pattern");
    EXPECT_EQ(result.errors[1].schemaUri, "https://example.com/remote.json");
    EXPECT_EQ(result.errors[2].schemaPath, "/patternProperties/^x~1~0/pattern");
    EXPECT_EQ(result.errors[2].instancePath, "/x~1~0tag");

    std::vector<std::future<bool>> workers;
    for (std::size_t worker = 0; worker < 4; ++worker)
    {
        workers.push_back(std::async(std::launch::async, [schema = moved, valid, invalid]
        {
            for (std::size_t iteration = 0; iteration < 16; ++iteration)
            {
                if (!schema.Validate(valid) || schema.Validate(invalid).errors.size() != 3)
                {
                    return false;
                }
            }
            return true;
        }));
    }
    for (auto& worker : workers)
    {
        EXPECT_TRUE(worker.get());
    }
}

#if !defined(RAD_JSON_SCHEMA_USE_STD_REGEX) || !RAD_JSON_SCHEMA_USE_STD_REGEX
TEST(IO, JsonSchemaRegexMatchLimits)
{
    constexpr std::string_view pattern = "^((((((((((a+))))))))))$";
    struct LimitCase
    {
        rad::JsonSchemaRegexMatchLimits limits;
        std::string_view message;
    };
    constexpr LimitCase limits[] = {
        {{0, 1'000, 8'192}, "match limit"},     {{1, 1'000, 8'192}, "match limit"},
        {{1'000'000, 0, 8'192}, "depth limit"}, {{1'000'000, 1, 8'192}, "depth limit"},
        {{1'000'000, 1'000, 0}, "heap limit"},  {{1'000'000, 1'000, 1}, "heap limit"},
    };
    struct SchemaCase
    {
        rad::JsonValue schema;
        rad::JsonValue instance;
        std::string_view path;
    };
    const SchemaCase schemas[] = {
        {rad::JsonObject{{"pattern", pattern}}, "aaaa", "/pattern"},
        {rad::JsonObject{{"patternProperties", rad::JsonObject{{pattern, true}}}},
         rad::JsonObject{{"aaaa", 1}}, "/patternProperties/^((((((((((a+))))))))))$"},
        {rad::JsonObject{{"anyOf", rad::JsonArray{rad::JsonObject{{"pattern", pattern}}, true}}},
         "aaaa", "/anyOf/0/pattern"},
        {rad::JsonObject{{"not", rad::JsonObject{{"pattern", pattern}}}}, "aaaa", "/not/pattern"},
        {rad::JsonObject{{"oneOf", rad::JsonArray{rad::JsonObject{{"pattern", pattern}}, true}}},
         "aaaa", "/oneOf/0/pattern"},
        {rad::JsonObject{{"if", rad::JsonObject{{"pattern", pattern}}}, {"then", true}}, "aaaa",
         "/if/pattern"},
        {rad::JsonObject{{"contains", rad::JsonObject{{"pattern", pattern}}}},
         rad::JsonArray{"aaaa"}, "/contains/pattern"},
    };
    rad::JsonSchemaCompileOptions compileOptions{.dialect = rad::JsonSchemaDialect::Draft2020_12};
    compileOptions.retrievalUri = "https://example.com/root.json";
    for (const auto& schema : schemas)
    {
        const auto compiled = rad::JsonSchema::Compile(schema.schema, compileOptions);
        ASSERT_TRUE(compiled) << compiled.error().message;
        for (const auto& error : compiled.value().Validate(schema.instance).errors)
        {
            EXPECT_EQ(error.errorCode, rad::JsonSchemaValidationErrorCode::SchemaViolation);
        }
        for (const auto& limit : limits)
        {
            SCOPED_TRACE(limit.message);
            SCOPED_TRACE(std::format("match={}, backtrackingDepth={}, heapKiB={}",
                                     limit.limits.matchLimit, limit.limits.backtrackingDepthLimit,
                                     limit.limits.heapLimitKiB));
            rad::JsonSchemaValidationOptions options;
            options.regex = limit.limits;
            options.maxErrors = 1;
            const auto result = compiled.value().Validate(schema.instance, options);
            EXPECT_FALSE(result);
            ASSERT_EQ(result.errors.size(), 1);
            EXPECT_EQ(result.errors.back().errorCode,
                      rad::JsonSchemaValidationErrorCode::RegexResourceLimitExceeded);
            EXPECT_NE(result.errors.back().message.find(limit.message),
                      std::string::npos);
            EXPECT_EQ(result.errors.back().schemaPath, schema.path);
            EXPECT_EQ(result.errors.back().schemaUri, compileOptions.retrievalUri);
        }
        for (const auto& error : compiled.value().Validate(schema.instance).errors)
        {
            EXPECT_EQ(error.errorCode, rad::JsonSchemaValidationErrorCode::SchemaViolation);
        }
    }

    const auto compiled = rad::JsonSchema::Compile(
        schemas[0].schema, {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_TRUE(compiled);
    std::vector<std::future<bool>> workers;
    for (std::size_t worker = 0; worker < 4; ++worker)
    {
        workers.push_back(
            std::async(std::launch::async,
                       [schema = compiled.value(), worker]
                       {
                           rad::JsonSchemaValidationOptions options;
                           const bool expected = worker % 2 == 0;
                           options.regex.matchLimit = expected ? 2'000'000 : 0;
                           for (std::size_t iteration = 0; iteration < 16; ++iteration)
                           {
                               if (static_cast<bool>(schema.Validate("aaaa", options)) != expected)
                               {
                                   return false;
                               }
                           }
                           return true;
                       }));
    }
    for (auto& worker : workers)
    {
        EXPECT_TRUE(worker.get());
    }
}
TEST(IO, JsonSchemaRegexEvaluationFailure)
{
    const auto compiled = rad::JsonSchema::Compile(
        rad::JsonObject{{"pattern", "."}}, {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_TRUE(compiled);
    const auto result = compiled.value().Validate(rad::JsonValue(std::string(1, '\xff')));
    EXPECT_FALSE(result);
    ASSERT_EQ(result.errors.size(), 1);
    EXPECT_EQ(result.errors.back().errorCode,
              rad::JsonSchemaValidationErrorCode::RegexEvaluationFailed);
    EXPECT_EQ(result.errors.back().schemaPath, "/pattern");
    EXPECT_NE(result.errors.back().message.find("regular expression matching failed"),
              std::string::npos);
}

TEST(IO, JsonSchemaErrorLimitSkipsRegexEvaluation)
{
    const auto compiled =
        rad::JsonSchema::Compile(rad::JsonObject{{"enum", rad::JsonArray{"different"}},
                                                 {"minLength", 5},
                                                 {"pattern", "^((((((((((a+))))))))))$"}},
                                 {.dialect = rad::JsonSchemaDialect::Draft2020_12});
    ASSERT_TRUE(compiled);
    for (const std::size_t maxErrors : {0U, 1U, 2U, 3U, 64U})
    {
        SCOPED_TRACE(maxErrors);
        rad::JsonSchemaValidationOptions options;
        options.maxErrors = maxErrors;
        options.regex.matchLimit = 0;
        const auto result = compiled.value().Validate("aaaa", options);
        EXPECT_FALSE(result);
        const std::size_t expectedErrors = maxErrors <= 1 ? 1 : maxErrors == 2 ? 2 : 3;
        ASSERT_EQ(result.errors.size(), expectedErrors);
        EXPECT_EQ(result.errors.front().errorCode,
                  rad::JsonSchemaValidationErrorCode::SchemaViolation);
        if (maxErrors > 2)
        {
            EXPECT_EQ(result.errors.back().errorCode,
                      rad::JsonSchemaValidationErrorCode::RegexResourceLimitExceeded);
            EXPECT_EQ(result.errors.back().schemaPath, "/pattern");
        }
        else
        {
            EXPECT_EQ(result.errors.back().errorCode,
                      rad::JsonSchemaValidationErrorCode::SchemaViolation);
        }
    }
}
#endif
