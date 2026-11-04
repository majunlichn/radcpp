#include <rad/Core/String.h>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <vector>

TEST(Core, StrSplit)
{
    EXPECT_EQ(rad::StrSplit("alpha,beta,gamma", ","),
              (std::vector<std::string>{"alpha", "beta", "gamma"}));
    EXPECT_EQ(rad::StrSplit("alpha,,gamma,", ","),
              (std::vector<std::string>{"alpha", "", "gamma", ""}));
    EXPECT_EQ(rad::StrSplit("alpha,,gamma,", ",", true),
              (std::vector<std::string>{"alpha", "gamma"}));
    EXPECT_EQ(rad::StrSplit("alpha;beta,gamma", ",;"),
              (std::vector<std::string>{"alpha", "beta", "gamma"}));
    EXPECT_EQ(rad::StrSplit(";alpha,,gamma;", ",;", true),
              (std::vector<std::string>{"alpha", "gamma"}));
    EXPECT_EQ(rad::StrSplit("alpha", ""), (std::vector<std::string>{"alpha"}));
}

TEST(Core, StrReplaceAll)
{
    EXPECT_EQ(rad::StrReplaceAll("alpha beta alpha", "alpha", "gamma"), "gamma beta gamma");
    EXPECT_EQ(rad::StrReplaceAll("alpha beta", "delta", "gamma"), "alpha beta");
    EXPECT_EQ(rad::StrReplaceAll("alpha beta", " beta", ""), "alpha");
    EXPECT_EQ(rad::StrReplaceAll("alpha", "", "gamma"), "alpha");
}

TEST(Core, StrEqual)
{
    EXPECT_TRUE(rad::StrEqual("alpha", "alpha"));
    EXPECT_FALSE(rad::StrEqual("alpha", "Alpha"));
    EXPECT_FALSE(rad::StrEqual("alpha", "alphabet"));

    const std::string embeddedNull{"alpha\0tail", 10};
    EXPECT_TRUE(rad::StrEqual(std::string_view{embeddedNull.data(), embeddedNull.size()},
                              std::string_view{"alpha\0tail", 10}));
}

TEST(Core, StrCaseEqual)
{
    EXPECT_TRUE(rad::StrCaseEqual("alpha", "Alpha"));
    EXPECT_TRUE(rad::StrCaseEqual("ALPHA", "alpha"));
    EXPECT_FALSE(rad::StrCaseEqual("alpha", "alphabet"));
    EXPECT_FALSE(rad::StrCaseEqual("alpha", "beta"));

    const std::string embeddedNull{"alpha\0tail", 10};
    EXPECT_TRUE(rad::StrCaseEqual(std::string_view{embeddedNull.data(), embeddedNull.size()},
                                  std::string_view{"ALPHA\0TAIL", 10}));
}

TEST(Core, StrCmp)
{
    EXPECT_EQ(rad::StrCmp("alpha", "alpha"), 0);
    EXPECT_GT(rad::StrCmp("alpha", "ALPHA"), 0);
    EXPECT_LT(rad::StrCmp("alpha", "beta"), 0);
    EXPECT_GT(rad::StrCmp("beta", "alpha"), 0);
    EXPECT_LT(rad::StrCmp("alpha", "alphabet"), 0);
    EXPECT_GT(rad::StrCmp("alphabet", "alpha"), 0);

    const std::string nonNullTerminated = "alphabet";
    EXPECT_EQ(rad::StrCmp(std::string_view{nonNullTerminated.data(), 5}, "alpha"), 0);

    const std::string embeddedNull{"alpha\0tail", 10};
    EXPECT_GT(rad::StrCmp(std::string_view{embeddedNull.data(), embeddedNull.size()}, "alpha"), 0);
}

TEST(Core, StrCaseCmp)
{
    EXPECT_EQ(rad::StrCaseCmp("alpha", "ALPHA"), 0);
    EXPECT_LT(rad::StrCaseCmp("alpha", "beta"), 0);
    EXPECT_GT(rad::StrCaseCmp("beta", "alpha"), 0);
    EXPECT_LT(rad::StrCaseCmp("alpha", "alphabet"), 0);
    EXPECT_GT(rad::StrCaseCmp("alphabet", "alpha"), 0);

    const std::string nonNullTerminated = "alphabet";
    EXPECT_EQ(rad::StrCaseCmp(std::string_view{nonNullTerminated.data(), 5}, "ALPHA"), 0);

    const std::string embeddedNull{"alpha\0tail", 10};
    EXPECT_EQ(rad::StrCaseCmp(std::string_view{embeddedNull.data(), embeddedNull.size()},
                              std::string_view{"ALPHA\0TAIL", 10}),
              0);
}

TEST(Core, StrUpper)
{
    EXPECT_EQ(rad::StrUpper("Alpha Beta 123!"), "ALPHA BETA 123!");
    EXPECT_EQ(rad::StrUpper("ALPHA"), "ALPHA");
    EXPECT_EQ(rad::StrUpper(""), "");
}

TEST(Core, StrLower)
{
    EXPECT_EQ(rad::StrLower("Alpha Beta 123!"), "alpha beta 123!");
    EXPECT_EQ(rad::StrLower("alpha"), "alpha");
    EXPECT_EQ(rad::StrLower(""), "");
}

TEST(Core, StrTrim)
{
    EXPECT_EQ(rad::StrTrim(" \talpha beta\r\n"), "alpha beta");
    EXPECT_EQ(rad::StrTrim("alpha  beta"), "alpha  beta");
    EXPECT_EQ(rad::StrTrim(" \t\r\n"), "");
    EXPECT_EQ(rad::StrTrim("alpha"), "alpha");
}

TEST(Core, RangeToString)
{
    EXPECT_EQ(rad::RangeToString(std::vector<int>{}), "");
    EXPECT_EQ(rad::RangeToString(std::vector<int>{7}), "7");
    EXPECT_EQ(rad::RangeToString(std::vector<int>{1, 2, 3}), "1, 2, 3");
    EXPECT_EQ(rad::RangeToString(std::vector<int>{1, 2, 3}, "|"), "1|2|3");
    EXPECT_EQ(rad::RangeToString(std::views::iota(0, 4)), "0, 1, 2, 3");

    EXPECT_EQ(rad::RangeToString(std::vector<int>{}, ", ", [](int x) { return x; }), "");
    EXPECT_EQ(rad::RangeToString(std::vector<int>{1, 2, 3}, ", ", [](int x) { return x * 10; }),
              "10, 20, 30");
    EXPECT_EQ(rad::RangeToString(std::vector<int>{1, 2, 3}, "|", [](int x) { return x; }), "1|2|3");
    EXPECT_EQ(rad::RangeToString(std::vector<int>{1, 2, 3}, ", ",
                                 [](int x) { return std::to_string(x * 10); }),
              "10, 20, 30");
}

TEST(Core, StrToValue)
{
    EXPECT_EQ(rad::StrToBool("1"), true);
    EXPECT_EQ(rad::StrToBool("true"), true);
    EXPECT_EQ(rad::StrToBool("TRUE"), true);
    EXPECT_EQ(rad::StrToBool("on"), true);
    EXPECT_EQ(rad::StrToBool(" On "), true);

    EXPECT_EQ(rad::StrToBool("0"), false);
    EXPECT_EQ(rad::StrToBool("false"), false);
    EXPECT_EQ(rad::StrToBool("FALSE"), false);
    EXPECT_EQ(rad::StrToBool("off"), false);
    EXPECT_EQ(rad::StrToBool("\toff\n"), false);

    EXPECT_EQ(rad::StrToBool(""), std::nullopt);
    EXPECT_EQ(rad::StrToBool("2"), std::nullopt);
    EXPECT_EQ(rad::StrToBool("yes"), std::nullopt);
    EXPECT_EQ(rad::StrToBool("ture"), std::nullopt);
}

TEST(Core, ToHexString)
{
    const std::array bytes = {std::byte{0x00}, std::byte{0x01}, std::byte{0xab}, std::byte{0xff}};
    EXPECT_EQ(rad::ToHexString(bytes), "0001abff");
    EXPECT_EQ(rad::ToHexString(bytes, rad::HexCase::Upper), "0001ABFF");
    EXPECT_EQ(rad::ToHexString({}), "");
}

TEST(String, IsEmailRFC5321)
{
    static_assert(noexcept(rad::IsEmailRFC5321("user@example.com")));

    constexpr struct
    {
        std::string_view email;
        bool valid;
    } cases[] = {
        {"user@example.com", true},
        {"User.Name+tag@EXAMPLE.COM", true},
        {"!#$%&'*+-/=?^_`{|}~@example.com", true},
        {"a@b", true},
        {"a@123.456", true},
        {"a@my-host.example", true},
        {"a@xn--bcher-kva.example", true},
        {"\"\"@example.com", true},
        {"\"a..b@c\"@example.com", true},
        {R"("a\"b\\c\ d"@example.com)", true},
        {"user@[0.0.0.0]", true},
        {"user@[255.255.255.255]", true},
        {"user@[001.002.003.004]", true},
        {"user@[IPv6:::]", true},
        {"user@[IPv6:::1]", true},
        {"user@[ipv6:2001:DB8::1]", true},
        {"user@[IPv6:1:2:3:4:5:6:7:8]", true},
        {"user@[IPv6:1:2:3:4:5:6:7::]", true},
        {"user@[IPv6:::ffff:192.0.2.1]", true},
        {"user@[IPv6:1:2:3:4:5:6:192.0.2.1]", true},
        {"user@[IPv6:1:2:3:4:5::192.0.2.1]", true},
        {"", false},
        {"user", false},
        {"@example.com", false},
        {"user@", false},
        {"user@@example.com", false},
        {".user@example.com", false},
        {"user.@example.com", false},
        {"user..name@example.com", false},
        {"user name@example.com", false},
        {"user@example.com ", false},
        {"user(comment)@example.com", false},
        {"User <user@example.com>", false},
        {"<user@example.com>", false},
        {"\"user@example.com", false},
        {"\"user\".name@example.com", false},
        {R"("user\"@example.com)", false},
        {"\"user\tname\"@example.com", false},
        {"user\r\n@example.com", false},
        {"\xC3\xA9@example.com", false},
        {"\"\xC3\xA9\"@example.com", false},
        {"user@exam\xC3\xA9.com", false},
        {"user@.example.com", false},
        {"user@example.com.", false},
        {"user@example..com", false},
        {"user@-example.com", false},
        {"user@example-.com", false},
        {"user@exam_ple.com", false},
        {"user@[]", false},
        {"user@[127.0.0.1", false},
        {"user@[127.0.0]", false},
        {"user@[256.0.0.1]", false},
        {"user@[0001.0.0.1]", false},
        {"user@[IPv6:]", false},
        {"user@[IPv6::]", false},
        {"user@[IPv6:::1::2]", false},
        {"user@[IPv6:1:2:3:4:5:6:7]", false},
        {"user@[IPv6:1:2:3:4:5:6:7:8:9]", false},
        {"user@[IPv6:1:2:3:4:5:6:7:8::]", false},
        {"user@[IPv6:::12345]", false},
        {"user@[IPv6:::gggg]", false},
        {"user@[IPv6:1:2:3:4:5:6::192.0.2.1]", false},
        {"user@[IPv6:::ffff:256.0.0.1]", false},
        {"user@[IPv6:::192.0.2.1:1]", false},
        {"user@[IPv6:fe80::1%eth0]", false},
        {"user@[IPvFuture:abc]", false},
    };
    for (const auto& test : cases)
    {
        EXPECT_EQ(rad::IsEmailRFC5321(test.email), test.valid) << test.email;
    }

    const std::string local(64, 'a');
    EXPECT_TRUE(rad::IsEmailRFC5321(local + "@example.com"));
    EXPECT_FALSE(rad::IsEmailRFC5321(local + "a@example.com"));
    EXPECT_TRUE(rad::IsEmailRFC5321("\"" + std::string(62, 'a') + "\"@example.com"));
    EXPECT_FALSE(rad::IsEmailRFC5321("\"" + std::string(63, 'a') + "\"@example.com"));

    const std::string label(63, 'b');
    EXPECT_TRUE(rad::IsEmailRFC5321("a@" + label + ".com"));
    EXPECT_FALSE(rad::IsEmailRFC5321("a@" + label + "b.com"));

    const std::string domain = label + "." + label + "." + std::string(61, 'c');
    const std::string email = local + "@" + domain;
    ASSERT_EQ(email.size(), 254);
    EXPECT_TRUE(rad::IsEmailRFC5321(email));
    EXPECT_FALSE(rad::IsEmailRFC5321(email + "c"));

    const std::string_view backing = "user@example.com.invalid";
    EXPECT_TRUE(rad::IsEmailRFC5321(backing.substr(0, 16)));
    EXPECT_FALSE(rad::IsEmailRFC5321(std::string_view{}));
    EXPECT_FALSE(rad::IsEmailRFC5321(std::string_view{"a\0b@example.com", 15}));
    EXPECT_FALSE(rad::IsEmailRFC5321(std::string_view{"a@example.com\0", 14}));
}

TEST(Core, StringLess)
{
    const rad::StringLess less;

    EXPECT_TRUE(less("alpha", "beta"));
    EXPECT_FALSE(less("alpha", "alpha"));
    EXPECT_FALSE(less("beta", "alpha"));

    const std::string highBit{static_cast<char>(0x80)};
    const std::string lowBit{static_cast<char>(0x7F)};
    EXPECT_EQ(less(lowBit, highBit), std::string_view{lowBit}.compare(highBit) < 0);
    EXPECT_EQ(less(highBit, lowBit), std::string_view{highBit}.compare(lowBit) < 0);

    std::set<std::string, rad::StringLess> values = {"alpha", "beta"};
    EXPECT_NE(values.find(std::string_view{"alpha"}), values.end());
}

TEST(Core, StringLessCaseInsensitive)
{
    const rad::StringLessCaseInsensitive less;

    EXPECT_TRUE(less("alpha", "beta"));
    EXPECT_FALSE(less("alpha", "ALPHA"));
    EXPECT_FALSE(less("ALPHA", "alpha"));

    std::set<std::string, rad::StringLessCaseInsensitive> values = {"alpha", "beta"};
    EXPECT_NE(values.find(std::string_view{"ALPHA"}), values.end());
}
