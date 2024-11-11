#include <rad/Core/String.h>
#include <gtest/gtest.h>

TEST(Core, String)
{
    EXPECT_EQ(rad::StrRemovePrefix("prefix.string.suffix", "prefix."), "string.suffix");
    EXPECT_EQ(rad::StrRemovePrefix("prefix.string.suffix", "string"), "prefix.string.suffix");
    EXPECT_EQ(rad::StrRemoveSuffix("prefix.string.suffix", ".suffix"), "prefix.string");
    EXPECT_EQ(rad::StrRemoveSuffix("prefix.string.suffix", "string"), "prefix.string.suffix");
    EXPECT_EQ(rad::StrRemoveTokenFront("prefix.string.suffix", "."), "string.suffix");
    EXPECT_EQ(rad::StrRemoveTokenFront("prefix.string.suffix", "-"), "prefix.string.suffix");
    EXPECT_EQ(rad::StrRemoveTokenBack("prefix.string.suffix", "."), "prefix.string");
    EXPECT_EQ(rad::StrRemoveTokenBack("prefix.string.suffix", "-"), "prefix.string.suffix");
}
