#include <radcpp/Core/String.h>

#include <gtest/gtest.h>

TEST(Core, String)
{
    std::string str = "rad.Core.String";
    std::string_view strView = "rad.Core.String";
    EXPECT_TRUE(rad::StrEqual(str, strView));
    EXPECT_FALSE(rad::StrEqual(str, "rad.Core"));
    EXPECT_FALSE(rad::StrEqual(str, "rad.Core.Memory"));

    EXPECT_TRUE(rad::StrCaseEqual(str, "rad.Core.String"));
    EXPECT_TRUE(rad::StrCaseEqual(str, rad::StrUpper(str)));
    EXPECT_TRUE(rad::StrCaseEqual(str, rad::StrLower(str)));

    EXPECT_TRUE(rad::StrTrim("").empty());
    EXPECT_TRUE(rad::StrTrim("    ").empty());
    EXPECT_TRUE(rad::StrTrim("==*****==", "=*").empty());
    EXPECT_EQ(rad::StrTrim("==**Title**==", "=*"), "Title");

    auto tokens = rad::StrSplit("rad.Core.String", ".", true);
    EXPECT_EQ(tokens.size(), 3);
    if (tokens.size() == 3)
    {
        EXPECT_EQ(tokens[0], "rad");
        EXPECT_EQ(tokens[1], "Core");
        EXPECT_EQ(tokens[2], "String");
    }
    auto views = rad::StrSplitViews("rad.Core.String", ".", true);
    EXPECT_EQ(views.size(), 3);
    if (views.size() == 3)
    {
        EXPECT_EQ(views[0], "rad");
        EXPECT_EQ(views[1], "Core");
        EXPECT_EQ(views[2], "String");
    }

    str = "rad.Core.String";
    str = rad::StrReplace(str, "Core", "System");
    rad::StrReplaceInPlace(str, "String", "Time");
    EXPECT_EQ(str, "rad.System.Time");
}
