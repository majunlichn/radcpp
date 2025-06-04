#pragma once

#include <SDFramework/Core/Common.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/Command.h>
#include <vkpp/Core/Fence.h>
#include <vkpp/Core/Semaphore.h>
#include <vkpp/Core/Buffer.h>
#include <vkpp/Core/Image.h>
#include <vkpp/Core/Sampler.h>
#include <vkpp/Core/Descriptor.h>
#include <vkpp/Core/Pipeline.h>

namespace sdf
{

rad::Ref<vkpp::Image> CreateTexture2DFromFile(
    vkpp::Device* device, std::string_view fileName, bool genMipmaps);

} // namespace sdf
