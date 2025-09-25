#pragma once

#include <rad/Common/Platform.h>
#include <boost/json.hpp>

namespace rad
{

using JsonValue = boost::json::value;
using JsonString = boost::json::string;
using JsonObject = boost::json::object;
using JsonArray = boost::json::array;

JsonValue ParseJson(std::string_view str);
JsonValue ParseJsonFromFile(std::string_view fileName);
JsonValue* FindMemberCaseInsensitive(JsonValue& value, std::string_view key);

} // namespace rad
