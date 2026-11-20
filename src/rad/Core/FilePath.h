#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace rad
{

using FilePath = std::filesystem::path;

// std::string paths are UTF-8 at this boundary and are converted to the platform-native encoding.
[[nodiscard]] FilePath MakeFilePath(std::string_view utf8Path);
[[nodiscard]] std::string ToUtf8(const FilePath& path);

} // namespace rad
