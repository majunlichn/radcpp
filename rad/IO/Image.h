#pragma once

#include <rad/Core/Platform.h>
#include <rad/Core/String.h>

namespace rad
{

// 8-bits unsigned-normalized data.
class ImageU8
{
public:
    ImageU8();
    ~ImageU8();

    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }
    int GetChannelCount() const { return m_channelCount; }

    bool Allocate(int width, int height, int channelCount);
    void Free();

    // Load image into 8-bits unorm format, gamma and scale for hdr to ldr convertion.
    // Please refer to: https://github.com/nothings/stb/blob/master/stb_image.h
    bool LoadFromFile(std::string_view filename, int channelCount, float gamma = 2.2f, float scale = 1.0f);

    // i in rows; j in colums.
    unsigned char* GetPixel(int i, int j)
    {
        size_t offset = (size_t(i) * size_t(m_width) + size_t(j)) *
            size_t(m_channelCount);
        return (m_data + offset);
    }

    void SetPixelR(int i, int j, unsigned r)
    {
        assert(m_channelCount == 1);
        GetPixel(i, j)[0] = static_cast<unsigned char>(r);
    }

    void SetPixelRG(int i, int j, unsigned r, unsigned g)
    {
        assert(m_channelCount == 2);
        GetPixel(i, j)[0] = static_cast<unsigned char>(r);
        GetPixel(i, j)[1] = static_cast<unsigned char>(g);
    }

    void SetPixelRGB(int i, int j, unsigned r, unsigned g, unsigned b)
    {
        assert(m_channelCount == 3);
        GetPixel(i, j)[0] = static_cast<unsigned char>(r);
        GetPixel(i, j)[1] = static_cast<unsigned char>(g);
        GetPixel(i, j)[2] = static_cast<unsigned char>(b);
    }

    void SetPixelRGBA(int i, int j, unsigned r, unsigned g, unsigned b, unsigned a)
    {
        assert(m_channelCount == 4);
        GetPixel(i, j)[0] = static_cast<unsigned char>(r);
        GetPixel(i, j)[1] = static_cast<unsigned char>(g);
        GetPixel(i, j)[2] = static_cast<unsigned char>(b);
        GetPixel(i, j)[3] = static_cast<unsigned char>(a);
    }

    bool WritePNG(std::string_view filename) const;
    // Write a sub-region.
    bool WritePNG(std::string_view filename, int left, int top, int right, int bottom) const;
    bool WriteBMP(std::string_view filename) const;
    bool WriteTGA(std::string_view filename) const;
    // JPEG ignores alpha channels in input data; quality is between 1 and 100.
    // Higher quality looks better but results in a bigger image.
    bool WriteJPG(std::string_view filename, int quality) const;

    unsigned char* m_data = nullptr;
    size_t m_sizeInBytes = 0;
    int m_width = 0;
    int m_height = 0;
    int m_channelCount = 0;

}; // class ImageU8

// Float32 HDR data.
class ImageFP32
{
public:
    ImageFP32();
    ~ImageFP32();

    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }
    int GetChannelCount() const { return m_channelCount; }

    bool Allocate(int width, int height, int channelCount);
    void Free();

    // Load image into (linear) floats to preserve the full dynamic range,
    // gamma and scale for ldr to hdr convertion.
    bool LoadFromFile(std::string_view filename, int channelCount, float gamma = 2.2f, float scale = 1.0f);

    float* GetPixel(int i, int j)
    {
        size_t offset = (size_t(i) * size_t(m_width) + size_t(j)) *
            size_t(m_channelCount);
        return (m_data + offset);
    }

    void SetPixelR(int i, int j, float r)
    {
        assert(m_channelCount == 1);
        GetPixel(i, j)[0] = r;
    }

    void SetPixelRG(int i, int j, float r, float g)
    {
        assert(m_channelCount == 2);
        GetPixel(i, j)[0] = r;
        GetPixel(i, j)[1] = g;
    }

    void SetPixelRGB(int i, int j, float r, float g, float b)
    {
        assert(m_channelCount == 3);
        GetPixel(i, j)[0] = r;
        GetPixel(i, j)[1] = g;
        GetPixel(i, j)[2] = b;
    }

    void SetPixelRGBA(int i, int j, float r, float g, float b, float a)
    {
        assert(m_channelCount == 4);
        GetPixel(i, j)[0] = r;
        GetPixel(i, j)[1] = g;
        GetPixel(i, j)[2] = b;
        GetPixel(i, j)[3] = a;
    }

    // HDR (radiance rgbE format) expects linear float data,
    // since the format is always 32-bit rgb(e) data, alpha (if provided) is discarded,
    // and for monochrome data it is replicated across all three channels.
    bool WriteHDR(std::string_view filename) const;

    float* m_data = nullptr;
    size_t m_sizeInBytes = 0;
    int m_width = 0;
    int m_height = 0;
    int m_channelCount = 0;

}; // class ImageFP32

} // namespace rad
