#include <rad/System/Application.h>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace
{

class TestEnvironment final : public testing::Environment
{
public:
    TestEnvironment(int argc, char** argv)
    {
        m_args.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index)
        {
            m_args.emplace_back(argv[index]);
        }
    }

    void SetUp() override
    {
        std::vector<char*> argv;
        argv.reserve(m_args.size());
        for (auto& arg : m_args)
        {
            argv.push_back(arg.data());
        }
        rad::Application::Instance().Init(static_cast<int>(argv.size()), argv.data());
    }

private:
    std::vector<std::string> m_args;
}; // class TestEnvironment

} // namespace

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new TestEnvironment(argc, argv));
    return RUN_ALL_TESTS();
}
