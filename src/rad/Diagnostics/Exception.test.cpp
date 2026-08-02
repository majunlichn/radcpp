#include <rad/Core/Platform.h>
#include <rad/Diagnostics/Exception.h>

#include <boost/exception/exception.hpp>
#include <boost/exception/info.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <string>
#include <type_traits>

namespace
{

RAD_NOINLINE rad::Exception MakeException()
{
    return rad::Exception{"test failure"};
}

class DerivedException final : public rad::Exception
{
public:
    DerivedException() :
        Exception("derived failure")
    {
    }
};

RAD_NOINLINE DerivedException MakeDerivedException()
{
    return {};
}

using TestContext = boost::error_info<struct TestContextTag, std::string>;

} // namespace

TEST(Diagnostics, Exception)
{
    static_assert(std::is_copy_constructible_v<rad::Exception>);

    const rad::Exception error = MakeException();

    EXPECT_STREQ(error.what(), "test failure");
    EXPECT_EQ(error.Message(), "test failure");
    EXPECT_NE(std::string{error.Location().file_name()}.find("Exception.test.cpp"),
              std::string::npos);
    EXPECT_NE(std::string{error.Location().function_name()}.find("MakeException"),
              std::string::npos);
    EXPECT_FALSE(error.StackTrace().empty());

    rad::Exception errorWithContext = error;
    errorWithContext << TestContext{"request 42"};
    const std::string diagnosticWithContext = errorWithContext.DiagnosticInformation();
    EXPECT_NE(diagnosticWithContext.find("request 42"), std::string::npos);

    const std::string diagnostic = error.DiagnosticInformation();
    EXPECT_NE(diagnostic.find("test failure"), std::string::npos);
    EXPECT_NE(diagnostic.find("Exception.test.cpp"), std::string::npos);
    EXPECT_EQ(diagnostic.find("Throw location unknown"), std::string::npos);
    EXPECT_NE(diagnostic.find("Stack trace:"), std::string::npos);

    EXPECT_NE(dynamic_cast<const std::exception*>(&error), nullptr);
    EXPECT_NE(dynamic_cast<const boost::exception*>(&error), nullptr);
}

TEST(Diagnostics, ExceptionCanDisableStackTrace)
{
    const rad::Exception error{"without trace", std::source_location::current(), 0};

    EXPECT_TRUE(error.StackTrace().empty());
    EXPECT_EQ(error.DiagnosticInformation().find("Stack trace:"), std::string::npos);
}

TEST(Diagnostics, ThrowExceptionCapturesDerivedThrowSite)
{
    const DerivedException exception = MakeDerivedException();
    std::uint_least32_t expectedLine = 0;
    try
    {
        expectedLine = std::source_location::current().line() + 1;
        rad::ThrowException(exception);
    }
    catch (const DerivedException& error)
    {
        EXPECT_EQ(error.Location().line(), expectedLine);
        EXPECT_NE(std::string{error.Location().function_name()}.find("TestBody"),
                  std::string::npos);
#if defined(_WIN32) && defined(_DEBUG)
        EXPECT_NE(error.StackTrace().find("TestBody"), std::string::npos);
#endif
        return;
    }

    FAIL() << "ThrowException did not throw";
}
