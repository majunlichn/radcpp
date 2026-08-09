#include <rad/Core/Flags.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

namespace rad
{

enum class TestFlagBits : std::uint8_t
{
    eRead = 0x01,
    eWrite = 0x02,
    eExecute = 0x04,
};

enum class PlainEnum : std::uint8_t
{
    eValue = 0x01,
};

template <>
struct FlagTraits<TestFlagBits>
{
    static constexpr bool isBitmask = true;
    static constexpr Flags<TestFlagBits> allFlags{
        static_cast<std::uint8_t>(TestFlagBits::eRead) |
        static_cast<std::uint8_t>(TestFlagBits::eWrite) |
        static_cast<std::uint8_t>(TestFlagBits::eExecute)};
};

} // namespace rad

namespace
{

using rad::TestFlagBits;
using TestFlags = rad::Flags<TestFlagBits>;

template <typename T>
concept FlagCompatible = requires { typename rad::Flags<T>; };

template <typename T>
concept HasComplement = requires(rad::Flags<T> flags) { ~flags; };

template <typename T>
concept HasFromMask =
    requires(typename rad::Flags<T>::Mask mask) { rad::Flags<T>::FromMask(mask); };

constexpr TestFlags readWrite = TestFlagBits::eRead | TestFlagBits::eWrite;
constexpr TestFlags fromMask = TestFlags::FromMask(0x03);
constexpr TestFlags invalidMask{0x80};

static_assert(!FlagCompatible<int>);
static_assert(rad::Bitmask<TestFlagBits>);
static_assert(!rad::Bitmask<rad::PlainEnum>);
static_assert(HasComplement<TestFlagBits>);
static_assert(!HasComplement<rad::PlainEnum>);
static_assert(HasFromMask<TestFlagBits>);
static_assert(!HasFromMask<rad::PlainEnum>);
static_assert(std::is_same_v<TestFlags::Bit, TestFlagBits>);
static_assert(std::is_same_v<TestFlags::Mask, std::uint8_t>);
static_assert(static_cast<TestFlags::Mask>(readWrite) == 0x03);
static_assert(fromMask == readWrite);
static_assert(fromMask.GetMask() == 0x03);
static_assert((~invalidMask).GetMask() == 0x07);
static_assert(readWrite == (TestFlagBits::eWrite | TestFlagBits::eRead));
static_assert((readWrite & TestFlagBits::eRead) == TestFlagBits::eRead);
static_assert((TestFlagBits::eRead & readWrite) == TestFlagBits::eRead);
static_assert((readWrite ^ TestFlagBits::eWrite) == TestFlagBits::eRead);
static_assert((~TestFlagBits::eRead) == (TestFlagBits::eWrite | TestFlagBits::eExecute));
static_assert(TestFlagBits::eRead < (TestFlagBits::eRead | TestFlagBits::eWrite));
static_assert(!readWrite.Empty());
static_assert(readWrite.HasAllBits(TestFlagBits::eRead));
static_assert(readWrite.HasAllBits(TestFlagBits::eRead | TestFlagBits::eWrite));
static_assert(!readWrite.HasAllBits(TestFlagBits::eExecute));
static_assert(readWrite.HasAnyBits(TestFlagBits::eWrite | TestFlagBits::eExecute));
static_assert(!readWrite.HasAnyBits(TestFlagBits::eExecute));
static_assert(readWrite.HasNoBits(TestFlagBits::eExecute));

TEST(Core, Flags)
{
    TestFlags flags;
    EXPECT_FALSE(flags);
    EXPECT_TRUE(!flags);
    EXPECT_TRUE(flags.Empty());

    flags |= TestFlagBits::eRead;
    flags |= TestFlagBits::eWrite;
    EXPECT_EQ(flags, readWrite);
    EXPECT_TRUE(flags.HasAllBits(TestFlagBits::eRead));
    EXPECT_TRUE(flags.HasAllBits(readWrite));
    EXPECT_TRUE(flags.HasAnyBits(TestFlagBits::eWrite | TestFlagBits::eExecute));
    EXPECT_TRUE(flags.HasNoBits(TestFlagBits::eExecute));

    flags &= TestFlagBits::eWrite;
    EXPECT_EQ(flags, TestFlagBits::eWrite);

    flags ^= TestFlagBits::eExecute;
    EXPECT_EQ(flags, TestFlagBits::eWrite | TestFlagBits::eExecute);
    EXPECT_EQ(flags.GetMask(), 0x06);
}

} // namespace
