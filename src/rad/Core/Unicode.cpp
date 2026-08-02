#include <rad/Core/Unicode.h>

#include <boost/locale/encoding_utf.hpp>

namespace rad
{
namespace
{

template <typename ToChar, typename FromChar>
std::basic_string<ToChar> UtfToUtf(std::basic_string_view<FromChar> value)
{
    if (value.empty())
    {
        return {};
    }

    return boost::locale::conv::utf_to_utf<ToChar>(value.data(), value.data() + value.size());
}

} // namespace

std::u16string Utf8ToUtf16(std::string_view value)
{
    return UtfToUtf<char16_t>(value);
}

std::u16string Utf8ToUtf16(std::u8string_view value)
{
    return UtfToUtf<char16_t>(value);
}

std::string Utf16ToUtf8(std::u16string_view value)
{
    return UtfToUtf<char>(value);
}

std::u8string Utf16ToUtf8Char8(std::u16string_view value)
{
    return UtfToUtf<char8_t>(value);
}

std::u32string Utf8ToUtf32(std::string_view value)
{
    return UtfToUtf<char32_t>(value);
}

std::u32string Utf8ToUtf32(std::u8string_view value)
{
    return UtfToUtf<char32_t>(value);
}

std::string Utf32ToUtf8(std::u32string_view value)
{
    return UtfToUtf<char>(value);
}

std::u8string Utf32ToUtf8Char8(std::u32string_view value)
{
    return UtfToUtf<char8_t>(value);
}

std::u32string Utf16ToUtf32(std::u16string_view value)
{
    return UtfToUtf<char32_t>(value);
}

std::u16string Utf32ToUtf16(std::u32string_view value)
{
    return UtfToUtf<char16_t>(value);
}

std::wstring Utf8ToWide(std::string_view value)
{
    return UtfToUtf<wchar_t>(value);
}

std::wstring Utf8ToWide(std::u8string_view value)
{
    return UtfToUtf<wchar_t>(value);
}

std::string WideToUtf8(std::wstring_view value)
{
    return UtfToUtf<char>(value);
}

std::u8string WideToUtf8Char8(std::wstring_view value)
{
    return UtfToUtf<char8_t>(value);
}

std::wstring Utf16ToWide(std::u16string_view value)
{
    return UtfToUtf<wchar_t>(value);
}

std::u16string WideToUtf16(std::wstring_view value)
{
    return UtfToUtf<char16_t>(value);
}

std::wstring Utf32ToWide(std::u32string_view value)
{
    return UtfToUtf<wchar_t>(value);
}

std::u32string WideToUtf32(std::wstring_view value)
{
    return UtfToUtf<char32_t>(value);
}

} // namespace rad
