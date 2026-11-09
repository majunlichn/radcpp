#include "JsonSchemaValues.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <bit>

namespace
{

[[nodiscard]] rad::JsonValue RandomValue(std::mt19937& random, std::size_t depth)
{
    switch (random() % (depth == 0 ? 6 : 8))
    {
    case 0:
        return nullptr;
    case 1:
        return random() % 2 != 0;
    case 2:
        return static_cast<std::int64_t>(random() % 33) - 16;
    case 3:
        return static_cast<std::uint64_t>(random() % 33);
    case 4:
        return (static_cast<double>(random() % 65) - 32) / 2;
    case 5:
        return rad::JsonValue(std::string("v\0", 2) + std::to_string(random() % 17));
    case 6:
    {
        rad::JsonArray array;
        const auto size = random() % 4;
        for (std::size_t index = 0; index < size; ++index)
        {
            array.push_back(RandomValue(random, depth - 1));
        }
        return array;
    }
    default:
    {
        rad::JsonObject object;
        const auto size = random() % 4;
        for (std::size_t index = 0; index < size; ++index)
        {
            object.emplace("key" + std::to_string(index), RandomValue(random, depth - 1));
        }
        return object;
    }
    }
}

[[nodiscard]] rad::JsonValue EquivalentValue(const rad::JsonValue& value)
{
    if (value.is_int64() || value.is_uint64())
    {
        const rad::JsonValue converted(value.is_int64() ? static_cast<double>(value.as_int64())
                                                       : static_cast<double>(value.as_uint64()));
        return rad::detail::JsonSchemaEqual(value, converted) ? converted : value;
    }
    if (value.is_array())
    {
        rad::JsonArray array;
        for (const auto& child : value.as_array())
        {
            array.push_back(EquivalentValue(child));
        }
        return array;
    }
    if (value.is_object())
    {
        rad::JsonObject object;
        const auto& original = value.as_object();
        for (std::size_t index = original.size(); index > 0; --index)
        {
            const auto& member = *(original.begin() + index - 1);
            object.emplace(member.key(), EquivalentValue(member.value()));
        }
        return object;
    }
    return value;
}

void ExpectDuplicateIndex(const rad::JsonArray& values, std::optional<std::size_t> index)
{
    if (!index)
    {
        return;
    }
    ASSERT_LT(*index, values.size());
    bool duplicate = false;
    for (std::size_t first = 0; first < *index; ++first)
    {
        duplicate = duplicate || rad::detail::JsonSchemaEqual(values[first], values[*index]);
    }
    EXPECT_TRUE(duplicate) << *index;
}

} // namespace

TEST(IO, JsonSchemaEquivalentValues)
{
    struct Case
    {
        rad::JsonValue value;
        bool doubleRepresentation;
    };
    const Case cases[] = {
        {std::int64_t{1}, true},
        {std::uint64_t{1}, true},
        {std::numeric_limits<std::int64_t>::min(), true},
        {std::numeric_limits<std::int64_t>::max(), false},
        {std::numeric_limits<std::uint64_t>::max(), false},
        {std::int64_t{9007199254740993LL}, false},
        {std::int64_t{-9007199254740993LL}, false},
        {std::uint64_t{9007199254740993ULL}, false},
        {std::uint64_t{9007199254740992ULL}, true},
        {std::uint64_t{9223372036854775808ULL}, true},
        {rad::JsonObject{
             {"small", 1},
             {"large", rad::JsonArray{std::numeric_limits<std::int64_t>::max(),
                                      std::numeric_limits<std::uint64_t>::max(),
                                      std::uint64_t{9007199254740993ULL}}}},
         false},
    };
    rad::detail::JsonSchemaValueHasher hash;
    for (const auto& testCase : cases)
    {
        SCOPED_TRACE(rad::PrettyJson(testCase.value));
        const auto equivalent = EquivalentValue(testCase.value);
        EXPECT_TRUE(rad::detail::JsonSchemaEqual(testCase.value, equivalent));
        EXPECT_EQ(equivalent.is_double(), testCase.doubleRepresentation);
        if (!testCase.doubleRepresentation)
        {
            EXPECT_EQ(equivalent.kind(), testCase.value.kind());
        }
        EXPECT_EQ(hash(equivalent), hash(testCase.value));
    }
}

TEST(IO, JsonSchemaValueHashNumbers)
{
    struct Case
    {
        rad::JsonValue left;
        rad::JsonValue right;
        bool equal;
    };
    const Case cases[] = {
        {std::int64_t{1}, std::uint64_t{1}, true},
        {std::int64_t{1}, 1.0, true},
        {std::uint64_t{1}, 1.0, true},
        {-0.0, std::uint64_t{0}, true},
        {0.0, -0.0, true},
        {std::numeric_limits<std::int64_t>::min(), -9223372036854775808.0, true},
        {std::numeric_limits<std::int64_t>::max(), 9223372036854775808.0, false},
        {std::numeric_limits<std::uint64_t>::max(), 18446744073709551616.0, false},
        {std::uint64_t{18446744073709549568ULL},
         std::nextafter(18446744073709551616.0, 0.0), true},
        {std::uint64_t{9007199254740993ULL}, 9007199254740992.0, false},
        {-1.0, std::int64_t{-1}, true},
        {-1.5, std::int64_t{-1}, false},
        {std::numeric_limits<double>::denorm_min(), 0.0, false},
        {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), true},
        {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), true},
        {-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), false},
        {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
         false},
    };
    rad::detail::JsonSchemaValueHasher hash;
    for (const auto& testCase : cases)
    {
        EXPECT_EQ(rad::detail::JsonSchemaEqual(testCase.left, testCase.right), testCase.equal);
        if (testCase.equal)
        {
            EXPECT_EQ(hash(testCase.left), hash(testCase.right));
        }
        const rad::JsonArray values{testCase.left, testCase.right};
        EXPECT_EQ(rad::detail::FindDuplicateIndexHashed(values).has_value(), testCase.equal);
    }
}

TEST(IO, JsonSchemaValueHashNumericRepresentations)
{
    std::mt19937_64 random(0xB17B0A4D);
    rad::detail::JsonSchemaValueHasher hash;
    const auto check = [&hash](const rad::JsonValue& left, const rad::JsonValue& right)
    {
        const bool equal = rad::detail::JsonSchemaEqual(left, right);
        if (equal)
        {
            EXPECT_EQ(hash(left), hash(right));
        }
        EXPECT_EQ(rad::detail::FindDuplicateIndexHashed(rad::JsonArray{left, right}).has_value(),
                  equal);
    };
    for (std::size_t iteration = 0; iteration < 4096; ++iteration)
    {
        SCOPED_TRACE(iteration);
        const auto unsignedInteger = random();
        const auto signedInteger = std::bit_cast<std::int64_t>(unsignedInteger);
        const auto number = std::bit_cast<double>(random());
        check(unsignedInteger, signedInteger);
        check(unsignedInteger, static_cast<double>(unsignedInteger));
        check(signedInteger, static_cast<double>(signedInteger));
        check(number, number);
    }
}

TEST(IO, JsonSchemaValueHashStructures)
{
    std::mt19937 random(0x5C4E4A);
    rad::detail::JsonSchemaValueHasher hash;
    for (std::size_t index = 0; index < 1000; ++index)
    {
        SCOPED_TRACE(index);
        const auto original = RandomValue(random, 4);
        const auto equivalent = EquivalentValue(original);
        ASSERT_TRUE(rad::detail::JsonSchemaEqual(original, equivalent));
        EXPECT_EQ(hash(original), hash(equivalent));
    }
    const rad::JsonValue original = rad::JsonObject{
        {"a", rad::JsonArray{1, rad::JsonObject{{"x", true}, {"y", nullptr}}}},
        {std::string("b\0c", 3), "value"}};
    EXPECT_EQ(hash(original), hash(EquivalentValue(original)));
    EXPECT_FALSE(rad::detail::FindDuplicateIndexHashed(
        rad::JsonArray{rad::JsonArray{1, 2}, rad::JsonArray{2, 1}}));
    EXPECT_FALSE(rad::detail::FindDuplicateIndexHashed(
        rad::JsonArray{rad::JsonObject{{"a", 1}, {"b", 2}},
                       rad::JsonObject{{"a", 2}, {"b", 1}}}));
}

TEST(IO, JsonSchemaValueHashDepth)
{
    rad::JsonValue left = 1;
    rad::JsonValue right = 1.0;
    for (std::size_t depth = 0; depth < 2048; ++depth)
    {
        const auto wrap = [depth](rad::JsonValue value) -> rad::JsonValue
        {
            if (depth % 2 == 0)
            {
                rad::JsonArray array;
                array.push_back(std::move(value));
                return array;
            }
            rad::JsonObject object;
            object.emplace("child", std::move(value));
            return object;
        };
        left = wrap(std::move(left));
        right = wrap(std::move(right));
    }
    rad::detail::JsonSchemaValueHasher hash;
    const auto leftHash = hash(left);
    const auto rightHash = hash(right);
    const auto unwrap = [](rad::JsonValue& value)
    {
        while (value.is_array() || value.is_object())
        {
            auto child = std::move(value.is_array() ? value.as_array()[0]
                                                   : value.as_object().begin()->value());
            value = std::move(child);
        }
    };
    unwrap(left);
    unwrap(right);
    EXPECT_EQ(leftHash, rightHash);
    EXPECT_EQ(hash(left), hash(right));
}

TEST(IO, JsonSchemaDuplicateHashCollisions)
{
    const auto collision = [](const rad::JsonValue&) -> std::uint64_t { return 0; };
    rad::JsonArray values;
    for (std::size_t index = 0; index < 128; ++index)
    {
        values.push_back(rad::JsonObject{{"index", index}, {"value", rad::JsonArray{1, 2}}});
    }
    EXPECT_FALSE(rad::detail::FindDuplicateIndexHashed(values, collision));
    values.push_back(EquivalentValue(values[0]));
    const auto duplicate = rad::detail::FindDuplicateIndexHashed(values, collision);
    ASSERT_TRUE(duplicate);
    EXPECT_EQ(*duplicate, values.size() - 1);
    ExpectDuplicateIndex(values, duplicate);

    values = {std::numeric_limits<double>::quiet_NaN(),
              std::numeric_limits<double>::quiet_NaN(), nullptr, false, 0, "0"};
    EXPECT_FALSE(rad::detail::FindDuplicateIndexHashed(values, collision));
    values.push_back(0.0);
    ExpectDuplicateIndex(values, rad::detail::FindDuplicateIndexHashed(values, collision));
    EXPECT_TRUE(rad::detail::FindDuplicateIndexHashed(values, collision));
}

TEST(IO, JsonSchemaDuplicateHashEarlyExit)
{
    constexpr auto threshold = rad::detail::JsonSchemaDuplicateHashThreshold;
    for (const auto size : {threshold, threshold + 1, threshold * 2 + 1, std::size_t{65536}})
    {
        SCOPED_TRACE(size);
        for (const auto repeated : {std::size_t{1}, std::size_t{2}, std::size_t{7},
                                    std::size_t{15}, threshold - 1, threshold, threshold + 1,
                                    threshold * 2})
        {
            if (repeated >= size)
            {
                continue;
            }
            SCOPED_TRACE(repeated);
            rad::JsonArray values;
            values.resize(size);
            for (std::size_t index = 0; index < repeated; ++index)
            {
                values[index] = index;
            }
            values[repeated] = 0;
            std::size_t visited = 0;
            rad::detail::JsonSchemaValueHasher hash;
            const auto countingHash = [&visited, &hash](const rad::JsonValue& value)
            {
                ++visited;
                return hash(value);
            };
            const auto duplicate = rad::detail::FindDuplicateIndexHashed(values, countingHash);
            ASSERT_TRUE(duplicate);
            EXPECT_EQ(*duplicate, repeated);
            EXPECT_EQ(visited, repeated + 1);
            EXPECT_EQ(rad::detail::FindDuplicateIndex(values), duplicate);
        }
    }
}

TEST(IO, JsonSchemaDuplicateHashLargeValues)
{
    rad::JsonArray values;
    for (std::size_t index = 0; index < 32; ++index)
    {
        rad::JsonArray array;
        array.resize(1024 + index);
        values.push_back(std::move(array));
        rad::JsonObject object;
        const auto properties = 16 + index;
        for (std::size_t property = 0; property < properties; ++property)
        {
            object.emplace("property" + std::to_string(property), property);
        }
        values.push_back(std::move(object));
        values.push_back(rad::JsonValue(std::string(4096 + index, 'x')));
    }
    std::size_t visited = 0;
    rad::detail::JsonSchemaValueHasher hash;
    const auto countingHash = [&visited, &hash](const rad::JsonValue& value)
    {
        ++visited;
        return hash(value);
    };
    EXPECT_FALSE(rad::detail::FindDuplicateIndexHashed(values, countingHash));
    EXPECT_EQ(visited, values.size());
    EXPECT_FALSE(rad::detail::FindDuplicateIndex(values));
    for (std::size_t kind = 0; kind < 3; ++kind)
    {
        SCOPED_TRACE(kind);
        values.push_back(EquivalentValue(values[kind]));
        visited = 0;
        const auto duplicate = rad::detail::FindDuplicateIndexHashed(values, countingHash);
        ASSERT_TRUE(duplicate);
        EXPECT_EQ(*duplicate, values.size() - 1);
        EXPECT_EQ(visited, values.size());
        ExpectDuplicateIndex(values, duplicate);
        values.pop_back();
    }
}

TEST(IO, JsonSchemaDuplicateAlgorithms)
{
    std::mt19937 random(0xD0911CA7);
    constexpr auto threshold = rad::detail::JsonSchemaDuplicateHashThreshold;
    for (const auto size : {std::size_t{0}, std::size_t{1}, threshold - 1, threshold,
                            threshold + 1, std::size_t{257}})
    {
        SCOPED_TRACE(size);
        for (std::size_t iteration = 0; iteration < 12; ++iteration)
        {
            rad::JsonArray values;
            for (std::size_t index = 0; index < size; ++index)
            {
                values.push_back(
                    rad::JsonObject{{"index", index}, {"value", RandomValue(random, 3)}});
            }
            const auto check = [&values]
            {
                const auto expected = rad::detail::FindDuplicateIndexPairwise(values);
                const auto hashed = rad::detail::FindDuplicateIndexHashed(values);
                const auto adaptive = rad::detail::FindDuplicateIndex(values);
                EXPECT_EQ(hashed.has_value(), expected.has_value());
                EXPECT_EQ(adaptive.has_value(), expected.has_value());
                ExpectDuplicateIndex(values, hashed);
                ExpectDuplicateIndex(values, adaptive);
            };
            check();
            if (size > 1)
            {
                values[1] = EquivalentValue(values.back());
                check();
            }
            if (size > 3)
            {
                values[size / 2] = EquivalentValue(values[0]);
                check();
            }
        }
    }
}
