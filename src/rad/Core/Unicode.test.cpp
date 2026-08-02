#include <rad/Core/Unicode.h>

#include <gtest/gtest.h>

#include <string>

TEST(Core, Unicode)
{
    const std::string str = "Hello, \xCF\x80 \xF0\x9F\x98\x80";
    const std::u8string utf8 = u8"Hello, \u03C0 \U0001F600";
    const std::u16string utf16 = u"Hello, \u03C0 \U0001F600";
    const std::u32string utf32 = U"Hello, \u03C0 \U0001F600";

    EXPECT_EQ(rad::Utf8ToUtf16(str), utf16);
    EXPECT_EQ(rad::Utf8ToUtf16(utf8), utf16);
    EXPECT_EQ(rad::Utf16ToUtf8(utf16), str);
    EXPECT_EQ(rad::Utf16ToUtf8Char8(utf16).compare(utf8), 0);

    EXPECT_EQ(rad::Utf8ToUtf32(str), utf32);
    EXPECT_EQ(rad::Utf8ToUtf32(utf8), utf32);
    EXPECT_EQ(rad::Utf32ToUtf8(utf32), str);
    EXPECT_EQ(rad::Utf32ToUtf8Char8(utf32).compare(utf8), 0);

    EXPECT_EQ(rad::Utf16ToUtf32(utf16), utf32);
    EXPECT_EQ(rad::Utf32ToUtf16(utf32), utf16);

    const std::wstring wide = rad::Utf8ToWide(str);
    EXPECT_EQ(rad::Utf8ToWide(utf8), wide);
    EXPECT_EQ(rad::WideToUtf8(wide), str);
    EXPECT_EQ(rad::WideToUtf8Char8(wide).compare(utf8), 0);
    EXPECT_EQ(rad::WideToUtf16(wide), utf16);
    EXPECT_EQ(rad::WideToUtf32(wide), utf32);

    EXPECT_EQ(rad::Utf16ToWide(utf16), wide);
    EXPECT_EQ(rad::Utf32ToWide(utf32), wide);

    EXPECT_TRUE(rad::Utf8ToUtf16("").empty());
    EXPECT_TRUE(rad::Utf8ToUtf16(std::u8string_view{}).empty());
    EXPECT_TRUE(rad::Utf16ToUtf8(u"").empty());
    EXPECT_TRUE(rad::Utf16ToUtf8Char8(u"").empty());
    EXPECT_TRUE(rad::Utf32ToUtf8(U"").empty());
    EXPECT_TRUE(rad::Utf32ToUtf8Char8(U"").empty());
    EXPECT_TRUE(rad::WideToUtf8(L"").empty());
    EXPECT_TRUE(rad::WideToUtf8Char8(L"").empty());
}
