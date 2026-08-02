#include <rad/IO/Image.h>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>
#include <stb_image_write.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace rad
{
namespace
{

[[nodiscard]] std::size_t CheckedValueCount(int width, int height, int channels)
{
    if ((width <= 0) || (height <= 0))
    {
        throw std::invalid_argument{"image width and height must be positive"};
    }
    if ((channels < 1) || (channels > 4))
    {
        throw std::invalid_argument{"image channel count must be between 1 and 4"};
    }
    if (width > INT_MAX / channels)
    {
        throw std::length_error{"image row is too large"};
    }

    const auto rowElements = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
    const auto imageHeight = static_cast<std::size_t>(height);
    if (rowElements > std::numeric_limits<std::size_t>::max() / imageHeight)
    {
        throw std::length_error{"image dimensions are too large"};
    }
    return rowElements * imageHeight;
}

[[nodiscard]] bool ValidateDesiredChannels(int desiredChannels) noexcept
{
    return desiredChannels >= 0 && desiredChannels <= 4;
}

[[nodiscard]] bool ValidateDimensions(int width, int height, int channels, int bytesPerValue = 1,
                                      int maxDimension = INT_MAX) noexcept
{
    if ((width <= 0) || (height <= 0) || (channels < 1) || (channels > 4) || (bytesPerValue <= 0) ||
        (width > maxDimension) || (height > maxDimension))
    {
        return false;
    }

    const std::int64_t rowValues = static_cast<std::int64_t>(width) * channels;
    return (rowValues <= INT_MAX / height) && (rowValues <= INT_MAX / bytesPerValue);
}

[[nodiscard]] bool ValidatePngDimensions(int width, int height, int channels) noexcept
{
    if (!ValidateDimensions(width, height, channels))
    {
        return false;
    }

    const std::int64_t filteredSize = (static_cast<std::int64_t>(width) * channels + 1) * height;
    return filteredSize <= INT_MAX;
}

[[nodiscard]] bool ValidateBmpDimensions(int width, int height, int channels) noexcept
{
    if (!ValidateDimensions(width, height, channels))
    {
        return false;
    }

    const int outputChannels = channels == 4 ? 4 : 3;
    const std::int64_t rowBytes =
        (static_cast<std::int64_t>(width) * outputChannels + 3) & ~std::int64_t{3};
    return rowBytes * height + 122 <= INT_MAX;
}

[[nodiscard]] bool ValidateHdrDimensions(int width, int height, int channels) noexcept
{
    return ValidateDimensions(width, height, channels, sizeof(float)) && width <= INT_MAX / 4;
}

void ValidateImage(int width, int height, int channels, std::size_t size)
{
    if (size == 0)
    {
        throw std::logic_error{"image is empty"};
    }
    if (CheckedValueCount(width, height, channels) != size)
    {
        throw std::logic_error{"image has inconsistent dimensions and storage"};
    }
}

[[nodiscard]] bool HasValidAlpha(const std::vector<float>& data, int channels) noexcept
{
    if ((channels != 2) && (channels != 4))
    {
        return true;
    }
    for (std::size_t index = static_cast<std::size_t>(channels - 1); index < data.size();
         index += static_cast<std::size_t>(channels))
    {
        const float alpha = data[index];
        if ((!std::isfinite(alpha)) || (alpha < 0.0f) || (alpha > 1.0f))
        {
            return false;
        }
    }
    return true;
}

[[nodiscard]] stbir_pixel_layout PixelLayout(int channels)
{
    switch (channels)
    {
    case 1:
        return STBIR_1CHANNEL;
    case 2:
        return STBIR_RA;
    case 3:
        return STBIR_RGB;
    case 4:
        return STBIR_RGBA;
    default:
        throw std::invalid_argument{"image channel count must be between 1 and 4"};
    }
}

} // namespace

ImageUnorm8::ImageUnorm8(int width, int height, int channels) :
    m_width(width),
    m_height(height),
    m_channels(channels),
    m_data(CheckedValueCount(width, height, channels))
{
}

ImageUnorm8::ImageUnorm8(int width, int height, int channels, std::vector<std::uint8_t> pixels) :
    m_width(width),
    m_height(height),
    m_channels(channels),
    m_data(std::move(pixels))
{
    if (CheckedValueCount(width, height, channels) != m_data.size())
    {
        throw std::invalid_argument{"pixel count does not match image dimensions"};
    }
}

std::optional<ImageUnorm8> ImageUnorm8::LoadFromFile(const std::string& fileName,
                                                     int desiredChannels)
{
    if (!ValidateDesiredChannels(desiredChannels))
    {
        return std::nullopt;
    }

    int width = 0;
    int height = 0;
    int sourceChannels = 0;
    using ImageData = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;
    ImageData pixels{stbi_load(fileName.c_str(), &width, &height, &sourceChannels, desiredChannels),
                     &stbi_image_free};
    if (!pixels)
    {
        return std::nullopt;
    }

    const int channels = desiredChannels != 0 ? desiredChannels : sourceChannels;
    const std::size_t valueCount = CheckedValueCount(width, height, channels);
    return ImageUnorm8{width, height, channels,
                       std::vector<std::uint8_t>{pixels.get(), pixels.get() + valueCount}};
}

std::optional<ImageUnorm8> ImageUnorm8::LoadFromMemory(const void* data, std::size_t size,
                                                       int desiredChannels)
{
    if ((!ValidateDesiredChannels(desiredChannels)) || (data == nullptr) || (size == 0) ||
        (size > static_cast<std::size_t>(INT_MAX)))
    {
        return std::nullopt;
    }

    int width = 0;
    int height = 0;
    int sourceChannels = 0;
    using ImageData = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;
    ImageData pixels{stbi_load_from_memory(static_cast<const stbi_uc*>(data),
                                           static_cast<int>(size), &width, &height, &sourceChannels,
                                           desiredChannels),
                     &stbi_image_free};
    if (!pixels)
    {
        return std::nullopt;
    }

    const int channels = desiredChannels != 0 ? desiredChannels : sourceChannels;
    const std::size_t valueCount = CheckedValueCount(width, height, channels);
    return ImageUnorm8{width, height, channels,
                       std::vector<std::uint8_t>{pixels.get(), pixels.get() + valueCount}};
}

bool ImageUnorm8::SavePNG(const std::string& fileName) const noexcept
{
    if ((Empty()) || (!ValidatePngDimensions(m_width, m_height, m_channels)))
    {
        return false;
    }
    const int stride = m_width * m_channels;
    return stbi_write_png(fileName.c_str(), m_width, m_height, m_channels, m_data.data(), stride) !=
           0;
}

bool ImageUnorm8::SavePNG(const std::string& fileName, int x, int y, int width,
                          int height) const noexcept
{
    if ((Empty()) || (x < 0) || (y < 0) || (width <= 0) || (height <= 0) || (x > m_width - width) ||
        (y > m_height - height) || (!ValidatePngDimensions(width, height, m_channels)))
    {
        return false;
    }
    const int stride = m_width * m_channels;
    return stbi_write_png(fileName.c_str(), width, height, m_channels, Pixel(x, y), stride) != 0;
}

bool ImageUnorm8::SaveJPEG(const std::string& fileName, int quality) const noexcept
{
    constexpr int MaximumDimension = UINT16_MAX;
    if ((Empty()) || ((m_channels != 1) && (m_channels != 3)) || (quality < 1) || (quality > 100) ||
        (!ValidateDimensions(m_width, m_height, m_channels, 1, MaximumDimension)))
    {
        return false;
    }
    return stbi_write_jpg(fileName.c_str(), m_width, m_height, m_channels, m_data.data(),
                          quality) != 0;
}

bool ImageUnorm8::SaveBMP(const std::string& fileName) const noexcept
{
    if ((Empty()) || (m_channels == 2) || (!ValidateBmpDimensions(m_width, m_height, m_channels)))
    {
        return false;
    }
    return stbi_write_bmp(fileName.c_str(), m_width, m_height, m_channels, m_data.data()) != 0;
}

bool ImageUnorm8::SaveTGA(const std::string& fileName) const noexcept
{
    constexpr int MaximumDimension = UINT16_MAX;
    if ((Empty()) || (!ValidateDimensions(m_width, m_height, m_channels, 1, MaximumDimension)))
    {
        return false;
    }
    return stbi_write_tga(fileName.c_str(), m_width, m_height, m_channels, m_data.data()) != 0;
}

bool ImageUnorm8::SaveHDR(const std::string& fileName) const noexcept
{
    try
    {
        return ToFloat32().SaveHDR(fileName);
    }
    catch (...)
    {
        return false;
    }
}

std::uint8_t* ImageUnorm8::Pixel(int x, int y) noexcept
{
    return const_cast<std::uint8_t*>(std::as_const(*this).Pixel(x, y));
}

const std::uint8_t* ImageUnorm8::Pixel(int x, int y) const noexcept
{
    assert(x >= 0 && x < m_width);
    assert(y >= 0 && y < m_height);
    return m_data.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(m_width) +
                            static_cast<std::size_t>(x)) *
                               static_cast<std::size_t>(m_channels);
}

ImageUnorm8 ImageUnorm8::Resize(int width, int height) const
{
    ValidateImage(m_width, m_height, m_channels, m_data.size());
    if ((!ValidateDimensions(m_width, m_height, m_channels)) ||
        (!ValidateDimensions(width, height, m_channels)))
    {
        throw std::length_error{"image dimensions are too large for stb_image_resize"};
    }
    ImageUnorm8 result{width, height, m_channels};
    if (!stbir_resize_uint8_linear(m_data.data(), m_width, m_height, 0, result.Data(), width,
                                   height, 0, PixelLayout(m_channels)))
    {
        throw std::runtime_error{"stb_image_resize failed to resize image"};
    }
    return result;
}

ImageFloat32 ImageUnorm8::ToFloat32() const
{
    ValidateImage(m_width, m_height, m_channels, m_data.size());
    std::vector<float> pixels(m_data.size());
    std::transform(m_data.begin(), m_data.end(), pixels.begin(),
                   [](std::uint8_t value) { return static_cast<float>(value) / 255.0f; });
    return ImageFloat32{m_width, m_height, m_channels, std::move(pixels)};
}

ImageFloat32::ImageFloat32(int width, int height, int channels) :
    m_width(width),
    m_height(height),
    m_channels(channels),
    m_data(CheckedValueCount(width, height, channels))
{
}

ImageFloat32::ImageFloat32(int width, int height, int channels, std::vector<float> pixels) :
    m_width(width),
    m_height(height),
    m_channels(channels),
    m_data(std::move(pixels))
{
    if (CheckedValueCount(width, height, channels) != m_data.size())
    {
        throw std::invalid_argument{"pixel count does not match image dimensions"};
    }
}

std::optional<ImageFloat32> ImageFloat32::LoadFromFile(const std::string& fileName,
                                                       int desiredChannels)
{
    if (!ValidateDesiredChannels(desiredChannels))
    {
        return std::nullopt;
    }
    if (!stbi_is_hdr(fileName.c_str()))
    {
        std::optional<ImageUnorm8> image = ImageUnorm8::LoadFromFile(fileName, desiredChannels);
        if (!image)
        {
            return std::nullopt;
        }
        return image->ToFloat32();
    }

    int width = 0;
    int height = 0;
    int sourceChannels = 0;
    using ImageData = std::unique_ptr<float, decltype(&stbi_image_free)>;
    ImageData pixels{
        stbi_loadf(fileName.c_str(), &width, &height, &sourceChannels, desiredChannels),
        &stbi_image_free};
    if (!pixels)
    {
        return std::nullopt;
    }

    const int channels = desiredChannels != 0 ? desiredChannels : sourceChannels;
    const std::size_t valueCount = CheckedValueCount(width, height, channels);
    return ImageFloat32{width, height, channels,
                        std::vector<float>{pixels.get(), pixels.get() + valueCount}};
}

std::optional<ImageFloat32> ImageFloat32::LoadFromMemory(const void* data, std::size_t size,
                                                         int desiredChannels)
{
    if ((!ValidateDesiredChannels(desiredChannels)) || (data == nullptr) || (size == 0) ||
        (size > static_cast<std::size_t>(INT_MAX)))
    {
        return std::nullopt;
    }
    if (!stbi_is_hdr_from_memory(static_cast<const stbi_uc*>(data), static_cast<int>(size)))
    {
        std::optional<ImageUnorm8> image = ImageUnorm8::LoadFromMemory(data, size, desiredChannels);
        if (!image)
        {
            return std::nullopt;
        }
        return image->ToFloat32();
    }

    int width = 0;
    int height = 0;
    int sourceChannels = 0;
    using ImageData = std::unique_ptr<float, decltype(&stbi_image_free)>;
    ImageData pixels{stbi_loadf_from_memory(static_cast<const stbi_uc*>(data),
                                            static_cast<int>(size), &width, &height,
                                            &sourceChannels, desiredChannels),
                     &stbi_image_free};
    if (!pixels)
    {
        return std::nullopt;
    }

    const int channels = desiredChannels != 0 ? desiredChannels : sourceChannels;
    const std::size_t valueCount = CheckedValueCount(width, height, channels);
    return ImageFloat32{width, height, channels,
                        std::vector<float>{pixels.get(), pixels.get() + valueCount}};
}

bool ImageFloat32::SavePNG(const std::string& fileName) const noexcept
{
    try
    {
        return ToUnorm8().SavePNG(fileName);
    }
    catch (...)
    {
        return false;
    }
}

bool ImageFloat32::SavePNG(const std::string& fileName, int x, int y, int width,
                           int height) const noexcept
{
    try
    {
        return ToUnorm8().SavePNG(fileName, x, y, width, height);
    }
    catch (...)
    {
        return false;
    }
}

bool ImageFloat32::SaveJPEG(const std::string& fileName, int quality) const noexcept
{
    try
    {
        return ToUnorm8().SaveJPEG(fileName, quality);
    }
    catch (...)
    {
        return false;
    }
}

bool ImageFloat32::SaveBMP(const std::string& fileName) const noexcept
{
    try
    {
        return ToUnorm8().SaveBMP(fileName);
    }
    catch (...)
    {
        return false;
    }
}

bool ImageFloat32::SaveTGA(const std::string& fileName) const noexcept
{
    try
    {
        return ToUnorm8().SaveTGA(fileName);
    }
    catch (...)
    {
        return false;
    }
}

bool ImageFloat32::SaveHDR(const std::string& fileName) const noexcept
{
    if ((Empty()) || ((m_channels != 1) && (m_channels != 3)) ||
        (!ValidateHdrDimensions(m_width, m_height, m_channels)))
    {
        return false;
    }
    constexpr float MaximumValue = 0x1p127f;
    if (std::ranges::any_of(m_data,
                            [](float value)
                            {
                                return std::isnan(value) || std::isinf(value) || (value < 0.0f) ||
                                       (value >= MaximumValue);
                            }))
    {
        return false;
    }
    return stbi_write_hdr(fileName.c_str(), m_width, m_height, m_channels, m_data.data()) != 0;
}

float* ImageFloat32::Pixel(int x, int y) noexcept
{
    return const_cast<float*>(std::as_const(*this).Pixel(x, y));
}

const float* ImageFloat32::Pixel(int x, int y) const noexcept
{
    assert(x >= 0 && x < m_width);
    assert(y >= 0 && y < m_height);
    return m_data.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(m_width) +
                            static_cast<std::size_t>(x)) *
                               static_cast<std::size_t>(m_channels);
}

ImageFloat32 ImageFloat32::Resize(int width, int height) const
{
    ValidateImage(m_width, m_height, m_channels, m_data.size());
    if ((!ValidateDimensions(m_width, m_height, m_channels, sizeof(float))) ||
        (!ValidateDimensions(width, height, m_channels, sizeof(float))))
    {
        throw std::length_error{"image dimensions are too large for stb_image_resize"};
    }
    if (!HasValidAlpha(m_data, m_channels))
    {
        throw std::invalid_argument{"image alpha values must be finite and between 0 and 1"};
    }
    ImageFloat32 result{width, height, m_channels};
    if (!stbir_resize_float_linear(m_data.data(), m_width, m_height, 0, result.Data(), width,
                                   height, 0, PixelLayout(m_channels)))
    {
        throw std::runtime_error{"stb_image_resize failed to resize image"};
    }
    return result;
}

ImageUnorm8 ImageFloat32::ToUnorm8() const
{
    ValidateImage(m_width, m_height, m_channels, m_data.size());
    std::vector<std::uint8_t> pixels(m_data.size());
    std::transform(m_data.begin(), m_data.end(), pixels.begin(),
                   [](float value)
                   {
                       const float clamped =
                           std::isnan(value) ? 0.0f : std::clamp(value, 0.0f, 1.0f);
                       return static_cast<std::uint8_t>(std::lround(clamped * 255.0f));
                   });
    return ImageUnorm8{m_width, m_height, m_channels, std::move(pixels)};
}

} // namespace rad
