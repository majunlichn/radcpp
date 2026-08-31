#include <rad/IO/File.h>

#include <rad/Core/String.h>

#include <gtest/gtest.h>

TEST(IO, File)
{
    // Test hash functions.
    {
        rad::FilePath path = "temp.bin";
        ASSERT_TRUE(rad::File::WriteText(path, "123456789"));
        EXPECT_EQ(rad::File::Crc32(path), 0xCBF43926U);
        EXPECT_EQ(rad::File::XxHash64(path), 0x8CB841DB40E6AE83ULL);
        const auto sha256 = rad::File::Sha256(path);
        ASSERT_TRUE(sha256);
        EXPECT_EQ(rad::ToHexString(*sha256),
                  "15e2b0d3c33891ebb0f1ef609ec419420c20e320ce94c65fbc8c3312448eb225");
        EXPECT_TRUE(rad::File::Remove(path));
    }
}
