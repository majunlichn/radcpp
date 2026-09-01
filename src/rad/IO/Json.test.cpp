#include <rad/IO/Json.h>

#include <boost/json.hpp>

#include <gtest/gtest.h>

#include <string>
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

TEST(IO, ParseJson)
{
    // Parse Non-Standard JSON with Comments and Trailing Commas
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
}
