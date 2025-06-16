#include <SDFramework/Rendering/VulkanTexture.h>

#include <compressonator.h>

namespace sdf
{

VkFormat ToVulkanFormat(CMP_FORMAT format);

rad::Ref<vkpp::Image> CreateTexture2DFromFile(vkpp::Device* device, std::string_view fileName, bool genMipmaps)
{
    rad::Ref<vkpp::Image> image;
    vk::Format format = vk::Format::eUndefined;
    CMP_MipSet mipSet = {};
    if (CMP_LoadTexture(fileName.data(), &mipSet) != CMP_OK)
    {
        SDF_LOG(err, "Failed to load texture: {}", fileName);
        return nullptr;
    }
    // For GPU, the texture must have width and height as a multiple of 4.
    if ((mipSet.m_nWidth % 4) > 0 || (mipSet.m_nHeight % 4) > 0)
    {
        SDF_LOG(err, "Cannot generate mipmap for image: {} (the size {}x{} is not multiple of 4)",
            fileName, mipSet.m_nWidth, mipSet.m_nHeight);
        genMipmaps = false;
    }

    format = static_cast<vk::Format>(ToVulkanFormat(mipSet.m_format));
    if (format != vk::Format::eUndefined)
    {
        if (genMipmaps && (mipSet.m_nMipLevels <= 1))
        {
            CMP_INT mipLevels = CMP_CalcMinMipSize(mipSet.m_nHeight, mipSet.m_nWidth, 16);
            CMP_GenerateMIPLevels(&mipSet, mipLevels);
        }

        image = device->CreateImage2D_Sampled(format,
            mipSet.m_nWidth, mipSet.m_nHeight, mipSet.m_nMipLevels);
        VkDeviceSize stagingBufferSize = VkDeviceSize(mipSet.dwDataSize) * 2;
        rad::Ref<vkpp::Buffer> stagingBuffer = vkpp::Buffer::CreateStagingUpload(device, stagingBufferSize);
        void* pStaging = stagingBuffer->MapMemory();
        CMP_DWORD offset = 0;
        for (CMP_INT level = 0; level < mipSet.m_nMipLevels; ++level)
        {
            CMP_MipLevel* levelData = nullptr;
            CMP_GetMipLevel(&levelData, &mipSet, level, 0);
            memcpy((uint8_t*)(pStaging)+offset,
                levelData->m_psbData, levelData->m_dwLinearSize);
            offset += levelData->m_dwLinearSize;
        }
        stagingBuffer->UnmapMemory();
        CopyBufferToImage2D(
            stagingBuffer.get(), 0,
            image.get(), 0, mipSet.m_nMipLevels, 0, 1);
        CMP_FreeMipSet(&mipSet);
    }
    return image;
}

VkFormat ToVulkanFormat(CMP_FORMAT format)
{
    switch (format)
    {
        // Channel Component formats
    case CMP_FORMAT_RGBA_8888_S:        return VK_FORMAT_R8G8B8A8_SNORM;
    case CMP_FORMAT_ARGB_8888_S:        return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_ARGB_8888:          return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_ABGR_8888:          return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_RGBA_8888:          return VK_FORMAT_R8G8B8A8_UNORM;
    case CMP_FORMAT_BGRA_8888:          return VK_FORMAT_B8G8R8A8_UNORM;
    case CMP_FORMAT_RGB_888:            return VK_FORMAT_R8G8B8_UNORM;
    case CMP_FORMAT_RGB_888_S:          return VK_FORMAT_R8G8B8_SNORM;
    case CMP_FORMAT_BGR_888:            return VK_FORMAT_B8G8R8_UNORM;
    case CMP_FORMAT_RG_8_S:             return VK_FORMAT_R8G8_SNORM;
    case CMP_FORMAT_RG_8:               return VK_FORMAT_R8G8_UNORM;
    case CMP_FORMAT_R_8_S:              return VK_FORMAT_R8_SNORM;
    case CMP_FORMAT_R_8:                return VK_FORMAT_R8_UNORM;
    case CMP_FORMAT_ARGB_2101010:       return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
    case CMP_FORMAT_RGBA_1010102:       return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_ARGB_16:            return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_ABGR_16:            return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_RGBA_16:            return VK_FORMAT_R16G16B16A16_UNORM;
    case CMP_FORMAT_BGRA_16:            return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_RG_16:              return VK_FORMAT_R16G16_UNORM;
    case CMP_FORMAT_R_16:               return VK_FORMAT_R16_UNORM;

    case CMP_FORMAT_RGBE_32F:           return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_ARGB_16F:           return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_ABGR_16F:           return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_RGBA_16F:           return VK_FORMAT_R16G16B16A16_SFLOAT;
    case CMP_FORMAT_BGRA_16F:           return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_RG_16F:             return VK_FORMAT_R16G16_SFLOAT;
    case CMP_FORMAT_R_16F:              return VK_FORMAT_R16_SFLOAT;
    case CMP_FORMAT_ARGB_32F:           return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_ABGR_32F:           return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_RGBA_32F:           return VK_FORMAT_R32G32B32A32_SFLOAT;
    case CMP_FORMAT_BGRA_32F:           return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_RGB_32F:            return VK_FORMAT_R32G32B32_SFLOAT;
    case CMP_FORMAT_BGR_32F:            return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_RG_32F:             return VK_FORMAT_R32G32_SFLOAT;
    case CMP_FORMAT_R_32F:              return VK_FORMAT_R32_SFLOAT;

        // Lossless Based Compression Formats
    case CMP_FORMAT_BROTLIG:            return VK_FORMAT_UNDEFINED;

        // Compression formats
    case CMP_FORMAT_BC1:                return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_BC2:                return VK_FORMAT_BC2_UNORM_BLOCK;
    case CMP_FORMAT_BC3:                return VK_FORMAT_BC3_UNORM_BLOCK;
    case CMP_FORMAT_BC4:                return VK_FORMAT_BC4_UNORM_BLOCK;
    case CMP_FORMAT_BC4_S:              return VK_FORMAT_BC4_SNORM_BLOCK;
    case CMP_FORMAT_BC5:                return VK_FORMAT_BC5_UNORM_BLOCK;
    case CMP_FORMAT_BC5_S:              return VK_FORMAT_BC5_SNORM_BLOCK;
    case CMP_FORMAT_BC6H:               return VK_FORMAT_BC6H_UFLOAT_BLOCK;
    case CMP_FORMAT_BC6H_SF:            return VK_FORMAT_BC6H_SFLOAT_BLOCK;
    case CMP_FORMAT_BC7:                return VK_FORMAT_BC7_UNORM_BLOCK;

    case CMP_FORMAT_ATI1N:              return VK_FORMAT_BC4_UNORM_BLOCK;
    case CMP_FORMAT_ATI2N:              return VK_FORMAT_BC5_UNORM_BLOCK;
    case CMP_FORMAT_ATI2N_XY:           return VK_FORMAT_BC5_UNORM_BLOCK;
    case CMP_FORMAT_ATI2N_DXT5:         return VK_FORMAT_BC5_UNORM_BLOCK;

    case CMP_FORMAT_DXT1:               return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
    case CMP_FORMAT_DXT3:               return VK_FORMAT_BC2_UNORM_BLOCK;

    case CMP_FORMAT_DXT5:               return VK_FORMAT_BC3_UNORM_BLOCK;
    case CMP_FORMAT_DXT5_xGBR:          return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_DXT5_RxBG:          return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_DXT5_RBxG:          return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_DXT5_xRBG:          return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_DXT5_RGxB:          return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_DXT5_xGxR:          return VK_FORMAT_UNDEFINED;

    case CMP_FORMAT_ATC_RGB:                return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_ATC_RGBA_Explicit:      return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_ATC_RGBA_Interpolated:  return VK_FORMAT_UNDEFINED;

    case CMP_FORMAT_ASTC:               return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
    case CMP_FORMAT_APC:                return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_PVRTC:              return VK_FORMAT_UNDEFINED;

    case CMP_FORMAT_ETC_RGB:            return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
    case CMP_FORMAT_ETC2_RGB:           return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
    case CMP_FORMAT_ETC2_SRGB:          return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
    case CMP_FORMAT_ETC2_RGBA:          return VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
    case CMP_FORMAT_ETC2_RGBA1:         return VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK;
    case CMP_FORMAT_ETC2_SRGBA:         return VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
    case CMP_FORMAT_ETC2_SRGBA1:        return VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK;

        // New Compression Formats -------------------------------------------------------------
    case CMP_FORMAT_BINARY:             return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_GTC:                return VK_FORMAT_UNDEFINED;
    case CMP_FORMAT_BASIS:              return VK_FORMAT_UNDEFINED;
    }

    return VK_FORMAT_UNDEFINED;
}

} // namespace sdf
