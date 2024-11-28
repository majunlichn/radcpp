#include <gtest/gtest.h>

#include <rfl/json.hpp>
#include <rfl.hpp>

TEST(Core, SimpleReflection)
{
    struct Person {
        std::string first_name;
        std::string last_name;
        int age;
    };

    const auto homer =
        Person{ .first_name = "Homer",
               .last_name = "Simpson",
               .age = 45 };

    // We can now write into and read from a JSON string.
    const std::string json_string = rfl::json::write(homer);
    auto homer2 = rfl::json::read<Person>(json_string).value();

    EXPECT_EQ(json_string, R"({"first_name":"Homer","last_name":"Simpson","age":45})");
    EXPECT_EQ(homer2.first_name, "Homer");
    EXPECT_EQ(homer2.last_name, "Simpson");
    EXPECT_EQ(homer2.age, 45);
}
