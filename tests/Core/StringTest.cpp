#include <rad/Core/String.h>
#include <gtest/gtest.h>

TEST(Core, String)
{
    EXPECT_EQ(rad::StrRemovePrefix("prefix.name.suffix", "prefix."), "name.suffix");
    EXPECT_EQ(rad::StrRemovePrefix("prefix.name.suffix", "name"), "prefix.name.suffix");
    EXPECT_EQ(rad::StrRemoveSuffix("prefix.name.suffix", ".suffix"), "prefix.name");
    EXPECT_EQ(rad::StrRemoveSuffix("prefix.name.suffix", "name"), "prefix.name.suffix");
    EXPECT_EQ(rad::StrRemovePrefixDelimited("prefix.name.suffix", "."), "name.suffix");
    EXPECT_EQ(rad::StrRemovePrefixDelimited("prefix.name.suffix", "#"), "prefix.name.suffix");
    EXPECT_EQ(rad::StrRemoveSuffixDelimited("prefix.name.suffix", "."), "prefix.name");
    EXPECT_EQ(rad::StrRemoveSuffixDelimited("prefix.name.suffix", "#"), "prefix.name.suffix");
}
