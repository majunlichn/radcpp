#include <rad/Core/Platform.h>
#include <rad/Diagnostics/StackTrace.h>

#include <gtest/gtest.h>

#include <string>

namespace
{

RAD_NOINLINE void DoSomething(std::size_t maxDepth = 32)
{
    const std::string trace = rad::GetStackTrace(maxDepth);

    EXPECT_FALSE(trace.empty());
#if defined(_WIN32) && defined(_DEBUG)
    EXPECT_NE(trace.find("DoSomething"), std::string::npos);
#endif
}

} // namespace

TEST(Diagnostics, StackTrace)
{
    DoSomething();
    DoSomething(1);

    EXPECT_TRUE(rad::GetStackTrace(0).empty());
}
