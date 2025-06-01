#include <vkpp/Core/Instance.h>
#include <vkpp/Core/Device.h>
#include <vkpp/Core/SlangSession.h>

#include <rad/IO/File.h>

#include <gtest/gtest.h>

extern rad::Ref<vkpp::Device> g_device;

TEST(Slang, HelloWorld)
{
    rad::Ref<vkpp::SlangSession> session = RAD_NEW vkpp::SlangSession();
    session->Init();
    slang::IModule* module = session->LoadModule("hello-world");

    session->CreateProgram({
        vkpp::SlangSession::ComponentDesc{ "hello-world", "computeMain" }
        });
    Slang::ComPtr<slang::IBlob> code = session->GenerateTargetCode();
    const void* buffer = code->getBufferPointer();
    size_t bufferSize = code->getBufferSize();
    EXPECT_GT(bufferSize, 0);
    if (bufferSize > 0)
    {
        EXPECT_TRUE(bufferSize % 4 == 0);   // SPIR-V code should be aligned to 4 bytes
        if (bufferSize > 4)
        {
            const uint32_t* spv = reinterpret_cast<const uint32_t*>(buffer);
            EXPECT_EQ(spv[0], 0x07230203); // check the magic number
        }
    }
}
