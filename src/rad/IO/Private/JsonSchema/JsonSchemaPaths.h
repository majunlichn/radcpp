#pragma once

#include <rad/IO/Json.h>

#include <string>
#include <string_view>

namespace rad::detail
{

[[nodiscard]] std::string ChildPath(std::string_view path, std::string_view token);
[[nodiscard]] const JsonValue* FindJsonSchemaValue(const JsonValue& root, std::string_view pointer);

} // namespace rad::detail
