#include <rad/Core/Base64.h>

#include <limits>
#include <stdexcept>

namespace rad
{
namespace
{

constexpr std::string_view Base64Alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

[[nodiscard]] constexpr int DecodeBase64Character(char value) noexcept
{
    if (value >= 'A' && value <= 'Z')
    {
        return value - 'A';
    }
    if (value >= 'a' && value <= 'z')
    {
        return value - 'a' + 26;
    }
    if (value >= '0' && value <= '9')
    {
        return value - '0' + 52;
    }
    if (value == '+')
    {
        return 62;
    }
    if (value == '/')
    {
        return 63;
    }
    return -1;
}

} // namespace

std::string EncodeBase64(Span<const std::byte> data)
{
    if (data.size() > std::numeric_limits<std::size_t>::max() / 4 * 3)
    {
        throw std::length_error("Base64 input is too large");
    }

    std::string encoded;
    encoded.reserve((data.size() / 3 + (data.size() % 3 != 0)) * 4);

    std::size_t offset = 0;
    while (data.size() - offset >= 3)
    {
        const auto first = std::to_integer<unsigned int>(data[offset]);
        const auto second = std::to_integer<unsigned int>(data[offset + 1]);
        const auto third = std::to_integer<unsigned int>(data[offset + 2]);

        encoded.push_back(Base64Alphabet[first >> 2]);
        encoded.push_back(Base64Alphabet[((first & 0x03u) << 4) | (second >> 4)]);
        encoded.push_back(Base64Alphabet[((second & 0x0Fu) << 2) | (third >> 6)]);
        encoded.push_back(Base64Alphabet[third & 0x3Fu]);
        offset += 3;
    }

    const auto remaining = data.size() - offset;
    if (remaining == 1)
    {
        const auto first = std::to_integer<unsigned int>(data[offset]);
        encoded.push_back(Base64Alphabet[first >> 2]);
        encoded.push_back(Base64Alphabet[(first & 0x03u) << 4]);
        encoded.append("==");
    }
    else if (remaining == 2)
    {
        const auto first = std::to_integer<unsigned int>(data[offset]);
        const auto second = std::to_integer<unsigned int>(data[offset + 1]);
        encoded.push_back(Base64Alphabet[first >> 2]);
        encoded.push_back(Base64Alphabet[((first & 0x03u) << 4) | (second >> 4)]);
        encoded.push_back(Base64Alphabet[(second & 0x0Fu) << 2]);
        encoded.push_back('=');
    }

    return encoded;
}

std::string EncodeBase64(std::string_view data)
{
    return EncodeBase64(AsBytes(data));
}

std::optional<std::vector<std::byte>> DecodeBase64(std::string_view data)
{
    if (data.size() % 4 != 0)
    {
        return std::nullopt;
    }

    std::vector<std::byte> decoded;
    decoded.reserve(data.size() / 4 * 3);

    for (std::size_t offset = 0; offset < data.size(); offset += 4)
    {
        const auto first = DecodeBase64Character(data[offset]);
        const auto second = DecodeBase64Character(data[offset + 1]);
        if (first < 0 || second < 0)
        {
            return std::nullopt;
        }

        const bool isLastGroup = offset + 4 == data.size();
        if (data[offset + 2] == '=')
        {
            if (!isLastGroup || data[offset + 3] != '=' || (second & 0x0F) != 0)
            {
                return std::nullopt;
            }

            decoded.push_back(static_cast<std::byte>((first << 2) | (second >> 4)));
            continue;
        }

        const auto third = DecodeBase64Character(data[offset + 2]);
        if (third < 0)
        {
            return std::nullopt;
        }

        decoded.push_back(static_cast<std::byte>((first << 2) | (second >> 4)));
        if (data[offset + 3] == '=')
        {
            if (!isLastGroup || (third & 0x03) != 0)
            {
                return std::nullopt;
            }

            decoded.push_back(static_cast<std::byte>(((second & 0x0F) << 4) | (third >> 2)));
            continue;
        }

        const auto fourth = DecodeBase64Character(data[offset + 3]);
        if (fourth < 0)
        {
            return std::nullopt;
        }

        decoded.push_back(static_cast<std::byte>(((second & 0x0F) << 4) | (third >> 2)));
        decoded.push_back(static_cast<std::byte>(((third & 0x03) << 6) | fourth));
    }

    return decoded;
}

} // namespace rad
