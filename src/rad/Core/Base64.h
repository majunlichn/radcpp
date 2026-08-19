#pragma once

#include <rad/Core/Span.h>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rad
{

// Encodes bytes using the standard RFC 4648 Section 4 alphabet with canonical '=' padding.
[[nodiscard]] std::string EncodeBase64(Span<const std::byte> data);
[[nodiscard]] std::string EncodeBase64(std::string_view data);

// Strictly decodes canonical RFC 4648 Section 4 Base64. Returns nullopt for invalid characters,
// whitespace, missing or misplaced padding, and non-zero unused trailing bits.
[[nodiscard]] std::optional<std::vector<std::byte>> DecodeBase64(std::string_view data);

} // namespace rad
