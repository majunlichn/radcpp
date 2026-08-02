#include <rad/IO/Logging.h>

#include <spdlog/sinks/ostream_sink.h>

#include <gtest/gtest.h>

#include <sstream>
#include <string>

TEST(IO, Logging)
{
    auto& logManager = rad::LogManager::Instance();
    logManager.Shutdown();
    EXPECT_FALSE(logManager.IsInitialized());
    logManager.Init();
    EXPECT_TRUE(logManager.IsInitialized());

    std::ostringstream output;
    logManager.ClearSinks();
    logManager.AddSink(std::make_shared<spdlog::sinks::ostream_sink_mt>(output));

    const auto logger = logManager.CreateLogger("LoggingTest");
    logger->info("Hello, World!");
    logger->flush();
    EXPECT_NE(output.str().find("Hello, World!"), std::string::npos);

    logManager.Shutdown();
    EXPECT_FALSE(logManager.IsInitialized());
}
