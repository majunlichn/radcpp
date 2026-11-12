#include <rad/IO/Json.h>

#include <boost/json.hpp>

#include <gtest/gtest.h>

#include <string_view>
#include <type_traits>
#include <utility>

static_assert(std::is_same_v<rad::JsonValue, boost::json::value>);
static_assert(std::is_same_v<rad::JsonObject, boost::json::object>);
static_assert(std::is_same_v<rad::JsonArray, boost::json::array>);
static_assert(std::is_same_v<rad::JsonString, boost::json::string>);
static_assert(std::is_same_v<rad::JsonKind, boost::json::kind>);
static_assert(std::is_same_v<rad::JsonErrorCode, boost::system::error_code>);
static_assert(std::is_same_v<rad::JsonParseOptions, boost::json::parse_options>);

TEST(Json, ParseWithCommentsAndTrailingCommas)
{
    constexpr std::string_view jsonString = R"json(
{
  // Comments and trailing commas are useful in configuration files.
  "name": "config",
  "values": [1, 2, 3,],
}
)json";

    rad::JsonParseOptions options;
    options.allow_comments = true;
    options.allow_trailing_commas = true;
    const auto json = rad::ParseJson(jsonString, options);
    ASSERT_TRUE(json) << json.error().message();
    EXPECT_EQ(json.value().as_object().at("values").as_array().size(), 3);
}

TEST(JsonDocument, ParseOptionsAndErrors)
{
    constexpr std::string_view text = R"json({/* comment */ "values": [1, 2,],})json";
    const auto strict = rad::JsonDocument::Parse(text);
    ASSERT_FALSE(strict);
    EXPECT_EQ(strict.error(), boost::json::error::syntax);

    rad::JsonParseOptions options;
    options.allow_comments = true;
    options.allow_trailing_commas = true;
    const auto parsed = rad::JsonDocument::Parse(text, options);
    ASSERT_TRUE(parsed) << parsed.error().message();
    EXPECT_EQ(parsed.value()->Root(), (rad::JsonObject{{"values", rad::JsonArray{1, 2}}}));

    const auto invalid = rad::JsonDocument::Parse("{");
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.error(), boost::json::error::incomplete);
}

TEST(JsonDocument, FindPointers)
{
    const auto owner = rad::JsonDocument::FromValue(
        rad::JsonObject{{"a/~", rad::JsonArray{1, rad::JsonObject{{"", true}}}}});
    auto& document = *owner;
    const auto element = document.Find("/a~1~0/0");
    ASSERT_TRUE(element) << element.error().message();
    EXPECT_EQ(element.value()->as_int64(), 1);
    const auto nested = std::as_const(document).Find("/a~1~0/1/");
    ASSERT_TRUE(nested) << nested.error().message();
    EXPECT_TRUE(nested.value()->as_bool());
    const auto root = document.Find("");
    ASSERT_TRUE(root);
    EXPECT_EQ(root.value(), &document.Root());

    const auto missing = document.Find("/missing");
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error(), boost::json::error::not_found);
    EXPECT_FALSE(document.Find("/a~1~0/01"));
    const auto invalidEscape = document.Find("/a~2");
    ASSERT_FALSE(invalidEscape);
    EXPECT_EQ(invalidEscape.error(), boost::json::error::invalid_escape);
    const auto invalidPointer = std::as_const(document).Find("invalid");
    ASSERT_FALSE(invalidPointer);
    EXPECT_EQ(invalidPointer.error(), boost::json::error::missing_slash);
}

TEST(JsonDocument, CopyOwnsIndependentValues)
{
    rad::JsonValue root = rad::JsonObject{{"values", rad::JsonArray{1, 2, 3}}};
    const auto document = rad::JsonDocument::FromValue(root);
    root.as_object().at("values").as_array()[0] = 42;
    root = nullptr;
    EXPECT_EQ(document->Root(), (rad::JsonObject{{"values", rad::JsonArray{1, 2, 3}}}));
}

TEST(JsonDocument, MovePreservesStorage)
{
    rad::JsonValue root = rad::JsonObject{{"values", rad::JsonArray{1, 2, 3}}};
    const auto* storage = root.as_object().at("values").as_array().data();
    const auto document = rad::JsonDocument::FromValue(std::move(root));
    EXPECT_EQ(document->Root().as_object().at("values").as_array().data(), storage);
}

TEST(JsonDocument, SetPointersAndLimits)
{
    const auto document = rad::JsonDocument::FromValue();
    ASSERT_TRUE(document->Set("/a~1~0/0", 1));
    ASSERT_TRUE(document->Set("/a~1~0/-", 2));
    const auto updated = document->Set("/a~1~0/0", 3);
    ASSERT_TRUE(updated) << updated.error().message();
    EXPECT_EQ(updated.value()->as_int64(), 3);
    EXPECT_EQ(document->Root().as_object().at("a/~").as_array(), (rad::JsonArray{3, 2}));
    EXPECT_FALSE(document->Set("/a~1~0/4", 5));
    EXPECT_EQ(document->Root().as_object().at("a/~").as_array(), (rad::JsonArray{3, 2}));

    rad::JsonSetPointerOptions options;
    options.max_created_elements = 3;
    ASSERT_TRUE(document->Set("/a~1~0/4", 5, options));
    EXPECT_EQ(document->Root().as_object().at("a/~").as_array(),
              (rad::JsonArray{3, 2, nullptr, nullptr, 5}));
    const auto invalidPointer = document->Set("invalid", 1);
    ASSERT_FALSE(invalidPointer);
    EXPECT_EQ(invalidPointer.error(), boost::json::error::missing_slash);
    const auto scalarParent = document->Set("/a~1~0/0/child", 1);
    ASSERT_FALSE(scalarParent);
    EXPECT_EQ(scalarParent.error(), boost::json::error::value_is_scalar);
    ASSERT_TRUE(document->Set("", false));
    EXPECT_FALSE(document->Root().as_bool());
}
