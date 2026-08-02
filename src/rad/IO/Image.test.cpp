#include <rad/IO/Image.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

[[nodiscard]] std::uint8_t GradientValue(int width, int height, int x, int y, int channel)
{
    const int widthRange = std::max(width - 1, 1);
    const int heightRange = std::max(height - 1, 1);
    switch (channel)
    {
    case 0:
        return static_cast<std::uint8_t>(x * 255 / widthRange);
    case 1:
        return static_cast<std::uint8_t>(y * 255 / heightRange);
    case 2:
        return static_cast<std::uint8_t>((widthRange - x + heightRange - y) * 255 /
                                         (widthRange + heightRange));
    default:
        return ((x / 32 + y / 32) % 2) != 0 ? 255 : 64;
    }
}

void FillGradient(rad::ImageUnorm8& image)
{
    for (int y = 0; y < image.Height(); ++y)
    {
        for (int x = 0; x < image.Width(); ++x)
        {
            std::uint8_t* pixel = image.Pixel(x, y);
            for (int channel = 0; channel < image.Channels(); ++channel)
            {
                pixel[channel] = GradientValue(image.Width(), image.Height(), x, y, channel);
            }
        }
    }
}

void VerifyImage(const rad::ImageUnorm8& expected, const rad::ImageUnorm8& actual, int tolerance)
{
    ASSERT_EQ(actual.Width(), expected.Width());
    ASSERT_EQ(actual.Height(), expected.Height());
    ASSERT_EQ(actual.Channels(), expected.Channels());

    for (int y = 0; y < expected.Height(); ++y)
    {
        for (int x = 0; x < expected.Width(); ++x)
        {
            for (int channel = 0; channel < expected.Channels(); ++channel)
            {
                EXPECT_LE(std::abs(static_cast<int>(actual.Pixel(x, y)[channel]) -
                                   static_cast<int>(expected.Pixel(x, y)[channel])),
                          tolerance);
            }
        }
    }
}

void VerifyRegion(const rad::ImageUnorm8& expected, const rad::ImageUnorm8& actual, int x, int y)
{
    ASSERT_LE(x + actual.Width(), expected.Width());
    ASSERT_LE(y + actual.Height(), expected.Height());
    ASSERT_EQ(actual.Channels(), expected.Channels());
    for (int row = 0; row < actual.Height(); ++row)
    {
        for (int column = 0; column < actual.Width(); ++column)
        {
            for (int channel = 0; channel < actual.Channels(); ++channel)
            {
                EXPECT_EQ(actual.Pixel(column, row)[channel],
                          expected.Pixel(x + column, y + row)[channel]);
            }
        }
    }
}

void TestLdrFormats()
{
    rad::ImageUnorm8 rgbaImage{256, 256, 4};
    FillGradient(rgbaImage);
    const std::string pngPath = "gradient.png";
    ASSERT_TRUE(rgbaImage.SavePNG(pngPath));
    const auto png = rad::ImageUnorm8::LoadFromFile(pngPath);
    ASSERT_TRUE(png);
    VerifyImage(rgbaImage, *png, 0);

    const std::string pngRegionPath = "gradient-region.png";
    ASSERT_TRUE(rgbaImage.SavePNG(pngRegionPath, 32, 64, 128, 96));
    const auto pngRegion = rad::ImageUnorm8::LoadFromFile(pngRegionPath);
    ASSERT_TRUE(pngRegion);
    ASSERT_EQ(pngRegion->Width(), 128);
    ASSERT_EQ(pngRegion->Height(), 96);
    VerifyRegion(rgbaImage, *pngRegion, 32, 64);

    rad::ImageUnorm8 rgbImage{256, 256, 3};
    FillGradient(rgbImage);

    const std::string bmpPath = "gradient.bmp";
    ASSERT_TRUE(rgbImage.SaveBMP(bmpPath));
    const auto bmp = rad::ImageUnorm8::LoadFromFile(bmpPath);
    ASSERT_TRUE(bmp);
    VerifyImage(rgbImage, *bmp, 0);

    const std::string tgaPath = "gradient.tga";
    ASSERT_TRUE(rgbImage.SaveTGA(tgaPath));
    const auto tga = rad::ImageUnorm8::LoadFromFile(tgaPath);
    ASSERT_TRUE(tga);
    VerifyImage(rgbImage, *tga, 0);

    const std::string jpegPath = "gradient.jpg";
    ASSERT_TRUE(rgbImage.SaveJPEG(jpegPath, 100));
    const auto jpeg = rad::ImageUnorm8::LoadFromFile(jpegPath);
    ASSERT_TRUE(jpeg);
    VerifyImage(rgbImage, *jpeg, 8);
}

void TestHdr()
{
    rad::ImageFloat32 image{256, 256, 3};
    for (int y = 0; y < image.Height(); ++y)
    {
        for (int x = 0; x < image.Width(); ++x)
        {
            const float horizontal = static_cast<float>(x) / static_cast<float>(image.Width() - 1);
            const float vertical = static_cast<float>(y) / static_cast<float>(image.Height() - 1);
            float* pixel = image.Pixel(x, y);
            pixel[0] = 0.25f + 3.75f * horizontal;
            pixel[1] = 0.25f + 3.75f * vertical;
            pixel[2] = 4.0f - 1.875f * (horizontal + vertical);
        }
    }

    const std::string path = "gradient.hdr";
    ASSERT_TRUE(image.SaveHDR(path));
    const auto loaded = rad::ImageFloat32::LoadFromFile(path);
    ASSERT_TRUE(loaded);
    ASSERT_EQ(loaded->Width(), image.Width());
    ASSERT_EQ(loaded->Height(), image.Height());
    ASSERT_EQ(loaded->Channels(), image.Channels());

    for (int y = 0; y < image.Height(); ++y)
    {
        for (int x = 0; x < image.Width(); ++x)
        {
            for (int channel = 0; channel < image.Channels(); ++channel)
            {
                EXPECT_NEAR(loaded->Pixel(x, y)[channel], image.Pixel(x, y)[channel], 0.04f);
            }
        }
    }
}

void TestFailures()
{
    rad::ImageUnorm8 rgbaImage{1, 1, 4};
    EXPECT_FALSE(rgbaImage.SavePNG(""));
    EXPECT_FALSE(rgbaImage.SaveJPEG("gradient-rgba.jpg"));

    rad::ImageFloat32 image{2, 2, 4};
    image.Pixel(0, 0)[3] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(static_cast<void>(image.Resize(1, 1)), std::invalid_argument);
}

} // namespace

TEST(IO, Image)
{
    TestLdrFormats();
    TestHdr();
    TestFailures();
}
