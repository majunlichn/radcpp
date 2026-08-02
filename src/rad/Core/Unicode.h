#pragma once

#include <string>
#include <string_view>

namespace rad
{

[[nodiscard]] std::u16string Utf8ToUtf16(std::string_view value);
[[nodiscard]] std::u16string Utf8ToUtf16(std::u8string_view value);
[[nodiscard]] std::string Utf16ToUtf8(std::u16string_view value);
[[nodiscard]] std::u8string Utf16ToUtf8Char8(std::u16string_view value);

[[nodiscard]] std::u32string Utf8ToUtf32(std::string_view value);
[[nodiscard]] std::u32string Utf8ToUtf32(std::u8string_view value);
[[nodiscard]] std::string Utf32ToUtf8(std::u32string_view value);
[[nodiscard]] std::u8string Utf32ToUtf8Char8(std::u32string_view value);

[[nodiscard]] std::u32string Utf16ToUtf32(std::u16string_view value);
[[nodiscard]] std::u16string Utf32ToUtf16(std::u32string_view value);

[[nodiscard]] std::wstring Utf8ToWide(std::string_view value);
[[nodiscard]] std::wstring Utf8ToWide(std::u8string_view value);
[[nodiscard]] std::string WideToUtf8(std::wstring_view value);
[[nodiscard]] std::u8string WideToUtf8Char8(std::wstring_view value);

[[nodiscard]] std::wstring Utf16ToWide(std::u16string_view value);
[[nodiscard]] std::u16string WideToUtf16(std::wstring_view value);

[[nodiscard]] std::wstring Utf32ToWide(std::u32string_view value);
[[nodiscard]] std::u32string WideToUtf32(std::wstring_view value);

} // namespace rad
