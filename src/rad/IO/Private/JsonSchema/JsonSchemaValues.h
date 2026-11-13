#pragma once

#include <rad/IO/Json.h>

#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>

namespace rad::detail
{

struct JsonSchemaDecimalNumber
{
    std::uint64_t coefficient;
    int exponent;
};

[[nodiscard]] bool IsNumber(const JsonValue& value) noexcept;
[[nodiscard]] long double AsNumber(const JsonValue& value) noexcept;
[[nodiscard]] std::partial_ordering CompareNumbers(const JsonValue& left,
                                                  const JsonValue& right) noexcept;
[[nodiscard]] bool JsonSchemaEqual(const JsonValue& left, const JsonValue& right);
// Unbounded recursive baseline for differential tests and profiling.
[[nodiscard]] bool JsonSchemaEqualRecursive(const JsonValue& left, const JsonValue& right);
[[nodiscard]] bool JsonSchemaEqualIterative(const JsonValue& left, const JsonValue& right);

class JsonSchemaValueHasher
{
public:
    [[nodiscard]] std::uint64_t operator()(const JsonValue& value);

private:
    struct Frame
    {
        const JsonValue* value;
        std::size_t next = 0;
        std::uint64_t hash = 0;
        std::uint64_t unorderedHash = 0;
    };

    std::vector<Frame> m_frames;
}; // class JsonSchemaValueHasher

inline constexpr std::size_t JsonSchemaDuplicateHashThreshold = 32;

[[nodiscard]] std::optional<std::size_t> FindDuplicateIndexPairwise(const JsonArray& values);

template <typename Hasher = JsonSchemaValueHasher>
[[nodiscard]] std::optional<std::size_t> FindDuplicateIndexHashed(const JsonArray& values,
                                                                Hasher hash = {})
{
    boost::unordered_flat_map<std::uint64_t, std::size_t> buckets;
    buckets.reserve(values.size());
    std::vector<std::size_t> previous;
    previous.reserve(values.size());
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        const auto [bucket, inserted] = buckets.try_emplace(hash(values[index]), index);
        if (!inserted)
        {
            // Fingerprints only select candidates; collisions must not imply equality.
            for (auto candidate = bucket->second; candidate != values.size();
                 candidate = previous[candidate])
            {
                if (JsonSchemaEqual(values[candidate], values[index]))
                {
                    return index;
                }
            }
        }
        previous.push_back(inserted ? values.size() : bucket->second);
        bucket->second = index;
    }
    return std::nullopt;
}

// Returns a repeated element's index; selection among multiple duplicates is unspecified.
[[nodiscard]] std::optional<std::size_t> FindDuplicateIndex(const JsonArray& values);

} // namespace rad::detail
