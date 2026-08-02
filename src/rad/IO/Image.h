#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rad
{

class ImageFloat32;

// An owning, tightly packed, row-major image with interleaved 8-bit UNORM channels.
// Channel layouts are Y, YA, RGB, and RGBA. No color-space conversion is performed.
class ImageUnorm8
{
public:
    ImageUnorm8() noexcept = default;
    ImageUnorm8(int width, int height, int channels);
    ImageUnorm8(int width, int height, int channels, std::vector<std::uint8_t> pixels);

    // desiredChannels must be 0 (preserve the source channel count) or 1 through 4.
    [[nodiscard]] static std::optional<ImageUnorm8> LoadFromFile(const std::string& fileName,
                                                                 int desiredChannels = 0);
    [[nodiscard]] static std::optional<ImageUnorm8> LoadFromMemory(const void* data,
                                                                   std::size_t size,
                                                                   int desiredChannels = 0);

    [[nodiscard]] bool SavePNG(const std::string& fileName) const noexcept;
    [[nodiscard]] bool SavePNG(const std::string& fileName, int x, int y, int width,
                               int height) const noexcept;
    // JPEG requires 1 or 3 channels. BMP accepts 1, 3, or 4 channels.
    [[nodiscard]] bool SaveJPEG(const std::string& fileName, int quality = 90) const noexcept;
    [[nodiscard]] bool SaveBMP(const std::string& fileName) const noexcept;
    [[nodiscard]] bool SaveTGA(const std::string& fileName) const noexcept;
    [[nodiscard]] bool SaveHDR(const std::string& fileName) const noexcept;

    // YA/RGBA input is straight alpha; colors are alpha-weighted during filtering.
    [[nodiscard]] ImageUnorm8 Resize(int width, int height) const;

    [[nodiscard]] ImageFloat32 ToFloat32() const;

    [[nodiscard]] int Width() const noexcept { return m_width; }
    [[nodiscard]] int Height() const noexcept { return m_height; }
    [[nodiscard]] int Channels() const noexcept { return m_channels; }
    [[nodiscard]] bool Empty() const noexcept { return m_data.empty(); }
    [[nodiscard]] std::uint8_t* Data() noexcept { return m_data.data(); }
    [[nodiscard]] const std::uint8_t* Data() const noexcept { return m_data.data(); }
    [[nodiscard]] std::uint8_t* Pixel(int x, int y) noexcept;
    [[nodiscard]] const std::uint8_t* Pixel(int x, int y) const noexcept;

private:
    int m_width = 0;
    int m_height = 0;
    int m_channels = 0;
    std::vector<std::uint8_t> m_data;
}; // class ImageUnorm8

// An owning, tightly packed, row-major image with interleaved 32-bit float channels.
// Channel layouts are Y, YA, RGB, and RGBA. No color-space conversion is performed.
// Alpha values used for resizing must be finite and in [0, 1].
class ImageFloat32
{
public:
    ImageFloat32() noexcept = default;
    ImageFloat32(int width, int height, int channels);
    ImageFloat32(int width, int height, int channels, std::vector<float> pixels);

    // desiredChannels must be 0 (preserve the source channel count) or 1 through 4.
    [[nodiscard]] static std::optional<ImageFloat32> LoadFromFile(const std::string& fileName,
                                                                  int desiredChannels = 0);
    [[nodiscard]] static std::optional<ImageFloat32> LoadFromMemory(const void* data,
                                                                    std::size_t size,
                                                                    int desiredChannels = 0);

    [[nodiscard]] bool SavePNG(const std::string& fileName) const noexcept;
    [[nodiscard]] bool SavePNG(const std::string& fileName, int x, int y, int width,
                               int height) const noexcept;
    [[nodiscard]] bool SaveJPEG(const std::string& fileName, int quality = 90) const noexcept;
    [[nodiscard]] bool SaveBMP(const std::string& fileName) const noexcept;
    [[nodiscard]] bool SaveTGA(const std::string& fileName) const noexcept;
    // Uses lossy RGBE encoding and requires 1 or 3 channels with finite, non-negative values.
    [[nodiscard]] bool SaveHDR(const std::string& fileName) const noexcept;

    // YA/RGBA input is straight alpha; colors are alpha-weighted during filtering.
    [[nodiscard]] ImageFloat32 Resize(int width, int height) const;

    // Values are clamped to [0, 1] and rounded to the nearest representable UNORM8 value.
    // NaN is converted to zero.
    [[nodiscard]] ImageUnorm8 ToUnorm8() const;

    [[nodiscard]] int Width() const noexcept { return m_width; }
    [[nodiscard]] int Height() const noexcept { return m_height; }
    [[nodiscard]] int Channels() const noexcept { return m_channels; }
    [[nodiscard]] bool Empty() const noexcept { return m_data.empty(); }
    [[nodiscard]] float* Data() noexcept { return m_data.data(); }
    [[nodiscard]] const float* Data() const noexcept { return m_data.data(); }
    [[nodiscard]] float* Pixel(int x, int y) noexcept;
    [[nodiscard]] const float* Pixel(int x, int y) const noexcept;

private:
    int m_width = 0;
    int m_height = 0;
    int m_channels = 0;
    std::vector<float> m_data;
}; // class ImageFloat32

} // namespace rad
