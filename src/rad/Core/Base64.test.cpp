#include <rad/Core/Base64.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

TEST(Core, EncodeBase64)
{
    EXPECT_EQ(rad::EncodeBase64(""), "");
    EXPECT_EQ(rad::EncodeBase64("f"), "Zg==");
    EXPECT_EQ(rad::EncodeBase64("fo"), "Zm8=");
    EXPECT_EQ(rad::EncodeBase64("foo"), "Zm9v");
    EXPECT_EQ(rad::EncodeBase64("foob"), "Zm9vYg==");
    EXPECT_EQ(rad::EncodeBase64("fooba"), "Zm9vYmE=");
    EXPECT_EQ(rad::EncodeBase64("foobar"), "Zm9vYmFy");

    const std::array binary = {
        std::byte{0x00}, std::byte{0xFF}, std::byte{0x10}, std::byte{0x80}, std::byte{0x7F},
    };
    EXPECT_EQ(rad::EncodeBase64(binary), "AP8QgH8=");
}

TEST(Core, DecodeBase64)
{
    const std::array vectors = {
        std::pair{std::string_view{""}, std::string_view{""}},
        std::pair{std::string_view{"Zg=="}, std::string_view{"f"}},
        std::pair{std::string_view{"Zm8="}, std::string_view{"fo"}},
        std::pair{std::string_view{"Zm9v"}, std::string_view{"foo"}},
        std::pair{std::string_view{"Zm9vYg=="}, std::string_view{"foob"}},
        std::pair{std::string_view{"Zm9vYmE="}, std::string_view{"fooba"}},
        std::pair{std::string_view{"Zm9vYmFy"}, std::string_view{"foobar"}},
    };

    for (const auto& [encoded, plain] : vectors)
    {
        const auto decoded = rad::DecodeBase64(encoded);
        ASSERT_TRUE(decoded);
        EXPECT_TRUE(std::ranges::equal(*decoded, rad::AsBytes(plain)));
    }

    const auto binary = rad::DecodeBase64("AP8QgH8=");
    ASSERT_TRUE(binary);
    EXPECT_EQ(*binary,
              (std::vector{
                  std::byte{0x00},
                  std::byte{0xFF},
                  std::byte{0x10},
                  std::byte{0x80},
                  std::byte{0x7F},
              }));
}

TEST(Core, DecodeBase64RejectsInvalidInput)
{
    constexpr std::array invalid = {
        std::string_view{"A"},        std::string_view{"AAA"},      std::string_view{"===="},
        std::string_view{"AA=A"},     std::string_view{"AA==AAAA"}, std::string_view{"AAA==="},
        std::string_view{"AA!?"},     std::string_view{"Zm 8="},    std::string_view{"Zh=="},
        std::string_view{"Zm9="},
    };

    for (const auto value : invalid)
    {
        EXPECT_FALSE(rad::DecodeBase64(value)) << value;
    }
}

TEST(Core, Base64RoundTrip)
{
    std::vector<std::byte> bytes;
    bytes.reserve(256);
    for (unsigned int value = 0; value < 256; ++value)
    {
        bytes.push_back(static_cast<std::byte>(value));
    }

    const auto decoded = rad::DecodeBase64(rad::EncodeBase64(bytes));
    ASSERT_TRUE(decoded);
    EXPECT_EQ(*decoded, bytes);
}
