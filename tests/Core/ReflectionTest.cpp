#include <rfl/json.hpp>
#include <rfl.hpp>

#include <rad/Core/TypeTraits.h>
#include <rad/Core/String.h>

#include <gtest/gtest.h>

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

TEST(Core, Reflection)
{
    // Age must be a plausible number, between 0 and 130.
    // This will be validated automatically.
    using Age = rfl::Validator<int, rfl::Minimum<0>, rfl::Maximum<130>>;

    struct Person {
        rfl::Rename<"firstName", std::string> first_name;
        rfl::Rename<"lastName", std::string> last_name = "Simpson";
        std::string town = "Springfield";
        rfl::Timestamp<"%Y-%m-%d"> birthday;
        Age age;
        rfl::Email email;
        std::vector<Person> children;
    };

    const auto bart = Person{ .first_name = "Bart",
                             .birthday = "1987-04-19",
                             .age = 10,
                             .email = "bart@simpson.com" };

    const auto lisa = Person{ .first_name = "Lisa",
                             .birthday = "1987-04-19",
                             .age = 8,
                             .email = "lisa@simpson.com" };

    const auto maggie = Person{ .first_name = "Maggie",
                               .birthday = "1987-04-19",
                               .age = 0,
                               .email = "maggie@simpson.com" };

    const auto homer =
        Person{ .first_name = "Homer",
               .birthday = "1987-04-19",
               .age = 45,
               .email = "homer@simpson.com",
               .children = std::vector<Person>({bart, lisa, maggie}) };

    // We can now transform this into a JSON string.
    const std::string json_string = rfl::json::write(homer);

    EXPECT_EQ(json_string, R"({"firstName":"Homer","lastName":"Simpson","town":"Springfield","birthday":"1987-04-19","age":45,"email":"homer@simpson.com","children":[{"firstName":"Bart","lastName":"Simpson","town":"Springfield","birthday":"1987-04-19","age":10,"email":"bart@simpson.com","children":[]},{"firstName":"Lisa","lastName":"Simpson","town":"Springfield","birthday":"1987-04-19","age":8,"email":"lisa@simpson.com","children":[]},{"firstName":"Maggie","lastName":"Simpson","town":"Springfield","birthday":"1987-04-19","age":0,"email":"maggie@simpson.com","children":[]}]})");

    auto homer2 = rfl::json::read<Person>(json_string).value();

    // Fields can be accessed like this:
    EXPECT_EQ(homer2.first_name(), "Homer");
    EXPECT_EQ(homer2.last_name(), "Simpson");

    // Since homer2 is mutable, we can also change the values like this:
    homer2.first_name = "Marge";

    EXPECT_EQ(homer2.first_name(), "Marge");
    EXPECT_EQ(homer2.last_name(), "Simpson");
}

// You cannot have more than 128 values and if you explicitly assign values, they must be between 0 and 127.
enum class Shape { square, circle };

// Flag enums: the important colors must be 1 or 2^N
enum class Color : int
{
    black = 0,
    red = 256,
    green = 512,
    blue = 1024,
    yellow = (256 | 512),
    purple = (256 | 1024),
    cyan = (512 | 1024),
    white = (256 | 512 | 1024),
};

// The bitwise OR operator must be defined - this is how reflect-cpp knows that
// this is a flag enum.
inline Color operator|(Color c1, Color c2) {
    return static_cast<Color>(static_cast<int>(c1) | static_cast<int>(c2));
}

TEST(Core, EnumReflection)
{
    struct Item {
        float pos_x;
        float pos_y;
        Shape shape;
        Color color;
    };

    const auto item = Item{
        .pos_x = 2.0f,
        .pos_y = 3.0f,
        .shape = Shape::square,
        .color = Color::white
    };

    std::string str = rfl::json::write(item);
    EXPECT_EQ(str, R"({"pos_x":2.0,"pos_y":3.0,"shape":"square","color":"red|green|blue"})");

    auto item2 = rfl::json::read<Item>(str).value();
    EXPECT_EQ(item2.pos_x, 2.0f);
    EXPECT_EQ(item2.pos_y, 3.0f);
    EXPECT_EQ(item2.shape, Shape::square);
    EXPECT_EQ(item2.color, Color::white);

    auto name = rfl::enum_to_string(Color::red);        // "red"
    EXPECT_EQ(name, "red");
    auto value = rfl::string_to_enum<Color>("red|green").value();
    EXPECT_EQ(value, Color::yellow);
}
