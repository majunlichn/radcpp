#include "JsonSchemaValues.h"

#include <rad/Core/Hash.h>

#include <bit>
#include <cmath>
#include <string_view>

namespace rad::detail
{
namespace
{

enum class HashTag : std::uint64_t
{
    Null,
    Boolean,
    String,
    NonNegativeInteger,
    NegativeInteger,
    Double,
    Array,
    Object,
};

[[nodiscard]] std::uint64_t Combine(std::uint64_t seed, std::uint64_t value) noexcept
{
    return pcg_hash64(seed ^ pcg_hash64(value));
}

[[nodiscard]] std::uint64_t TaggedHash(HashTag tag, std::uint64_t value) noexcept
{
    return Combine(static_cast<std::uint64_t>(tag), value);
}

[[nodiscard]] std::uint64_t HashString(std::string_view value) noexcept
{
    return XXH3_64bits_withSeed(value.data(), value.size(),
                               static_cast<std::uint64_t>(HashTag::String));
}

[[nodiscard]] std::uint64_t HashSignedInteger(std::int64_t value) noexcept
{
    if (value >= 0)
    {
        return TaggedHash(HashTag::NonNegativeInteger, static_cast<std::uint64_t>(value));
    }
    const auto magnitude = static_cast<std::uint64_t>(-(value + 1)) + 1;
    return TaggedHash(HashTag::NegativeInteger, magnitude);
}

[[nodiscard]] std::uint64_t HashNumber(const JsonValue& value) noexcept
{
    if (value.is_int64())
    {
        return HashSignedInteger(value.as_int64());
    }
    if (value.is_uint64())
    {
        return TaggedHash(HashTag::NonNegativeInteger, value.as_uint64());
    }
    const auto number = value.as_double();
    // Range checks precede conversions, including at the rounded int64/uint64 upper bounds.
    if (std::trunc(number) == number)
    {
        if (number >= 0 && number < 18446744073709551616.0)
        {
            return TaggedHash(HashTag::NonNegativeInteger, static_cast<std::uint64_t>(number));
        }
        if (number < 0 && number >= -9223372036854775808.0)
        {
            return HashSignedInteger(static_cast<std::int64_t>(number));
        }
    }
    return TaggedHash(HashTag::Double, std::bit_cast<std::uint64_t>(number));
}

[[nodiscard]] std::uint64_t HashObject(std::size_t size, std::uint64_t sum,
                                     std::uint64_t unorderedHash) noexcept
{
    return Combine(Combine(TaggedHash(HashTag::Object, size), sum), unorderedHash);
}

} // namespace

bool IsNumber(const JsonValue& value) noexcept
{
    return value.is_int64() || value.is_uint64() || value.is_double();
}

long double AsNumber(const JsonValue& value) noexcept
{
    if (value.is_int64())
    {
        return static_cast<long double>(value.as_int64());
    }
    if (value.is_uint64())
    {
        return static_cast<long double>(value.as_uint64());
    }
    return static_cast<long double>(value.as_double());
}

std::partial_ordering CompareNumbers(const JsonValue& left, const JsonValue& right) noexcept
{
    if ((left.is_double() && std::isnan(left.as_double())) ||
        (right.is_double() && std::isnan(right.as_double())))
    {
        return std::partial_ordering::unordered;
    }
    if (left.is_int64() && right.is_int64())
    {
        return left.as_int64() <=> right.as_int64();
    }
    if (left.is_uint64() && right.is_uint64())
    {
        return left.as_uint64() <=> right.as_uint64();
    }
    if (left.is_int64() && right.is_uint64())
    {
        if (left.as_int64() < 0)
        {
            return std::partial_ordering::less;
        }
        const auto converted = static_cast<std::uint64_t>(left.as_int64());
        return converted <=> right.as_uint64();
    }
    if (left.is_uint64() && right.is_int64())
    {
        return 0 <=> CompareNumbers(right, left);
    }
    if (left.is_int64() && right.is_double())
    {
        const double number = right.as_double();
        constexpr double lowerBound = -9223372036854775808.0;
        constexpr double upperBound = 9223372036854775808.0;
        if (number < lowerBound)
        {
            return std::partial_ordering::greater;
        }
        if (number >= upperBound)
        {
            return std::partial_ordering::less;
        }
        const auto integer = static_cast<std::int64_t>(number);
        if (left.as_int64() != integer)
        {
            return left.as_int64() <=> integer;
        }
        return static_cast<double>(integer) <=> number;
    }
    if (left.is_uint64() && right.is_double())
    {
        const double number = right.as_double();
        constexpr double upperBound = 18446744073709551616.0;
        if (number < 0)
        {
            return std::partial_ordering::greater;
        }
        if (number >= upperBound)
        {
            return std::partial_ordering::less;
        }
        const auto integer = static_cast<std::uint64_t>(number);
        if (left.as_uint64() != integer)
        {
            return left.as_uint64() <=> integer;
        }
        return static_cast<double>(integer) <=> number;
    }
    if (left.is_double() && !right.is_double())
    {
        return 0 <=> CompareNumbers(right, left);
    }

    const auto leftNumber = AsNumber(left);
    const auto rightNumber = AsNumber(right);
    return leftNumber <=> rightNumber;
}

bool JsonSchemaEqual(const JsonValue& lhs, const JsonValue& rhs)
{
    if (IsNumber(lhs) && IsNumber(rhs))
    {
        return CompareNumbers(lhs, rhs) == 0;
    }
    if (lhs.kind() != rhs.kind())
    {
        return false;
    }
    if (lhs.is_array())
    {
        const auto& lhsArray = lhs.as_array();
        const auto& rhsArray = rhs.as_array();
        if (lhsArray.size() != rhsArray.size())
        {
            return false;
        }
        for (std::size_t index = 0; index < lhsArray.size(); ++index)
        {
            if (!JsonSchemaEqual(lhsArray[index], rhsArray[index]))
            {
                return false;
            }
        }
        return true;
    }
    if (lhs.is_object())
    {
        const auto& lhsObject = lhs.as_object();
        const auto& rhsObject = rhs.as_object();
        if (lhsObject.size() != rhsObject.size())
        {
            return false;
        }
        for (const auto& member : lhsObject)
        {
            const auto* rhsValue = rhsObject.if_contains(member.key());
            if (rhsValue == nullptr || !JsonSchemaEqual(member.value(), *rhsValue))
            {
                return false;
            }
        }
        return true;
    }
    return lhs == rhs;
}

std::uint64_t JsonSchemaValueHasher::operator()(const JsonValue& root)
{
    m_frames.clear();
    const JsonValue* value = &root;
    for (;;)
    {
        std::uint64_t hash;
        if (value->is_array())
        {
            const auto& array = value->as_array();
            hash = TaggedHash(HashTag::Array, array.size());
            if (!array.empty())
            {
                m_frames.push_back({value, 0, hash});
                value = &array[0];
                continue;
            }
        }
        else if (value->is_object())
        {
            const auto& object = value->as_object();
            hash = HashObject(0, 0, 0);
            if (!object.empty())
            {
                m_frames.push_back({value});
                value = &object.begin()->value();
                continue;
            }
        }
        else if (value->is_string())
        {
            const auto& string = value->as_string();
            hash = HashString({string.data(), string.size()});
        }
        else if (value->is_bool())
        {
            hash = TaggedHash(HashTag::Boolean, value->as_bool());
        }
        else if (IsNumber(*value))
        {
            hash = HashNumber(*value);
        }
        else
        {
            hash = TaggedHash(HashTag::Null, 0);
        }

        for (;;)
        {
            if (m_frames.empty())
            {
                return hash;
            }
            auto& frame = m_frames.back();
            if (frame.value->is_array())
            {
                const auto& array = frame.value->as_array();
                frame.hash = Combine(frame.hash, hash);
                if (++frame.next < array.size())
                {
                    value = &array[frame.next];
                    break;
                }
                hash = frame.hash;
            }
            else
            {
                const auto& object = frame.value->as_object();
                const auto& member = *(object.begin() + frame.next);
                const auto memberHash = Combine(HashString(member.key()), hash);
                // Both reductions are commutative; member iteration order is not significant.
                frame.hash += memberHash;
                frame.unorderedHash ^= pcg_hash64(memberHash);
                if (++frame.next < object.size())
                {
                    value = &(object.begin() + frame.next)->value();
                    break;
                }
                hash = HashObject(object.size(), frame.hash, frame.unorderedHash);
            }
            m_frames.pop_back();
        }
    }
}

std::optional<std::size_t> FindDuplicateIndexPairwise(const JsonArray& values)
{
    for (std::size_t first = 0; first < values.size(); ++first)
    {
        for (std::size_t second = first + 1; second < values.size(); ++second)
        {
            if (JsonSchemaEqual(values[first], values[second]))
            {
                return second;
            }
        }
    }
    return std::nullopt;
}

std::optional<std::size_t> FindDuplicateIndex(const JsonArray& values)
{
    if (values.size() < JsonSchemaDuplicateHashThreshold)
    {
        return FindDuplicateIndexPairwise(values);
    }
    if (JsonSchemaEqual(values[0], values[1]))
    {
        return 1;
    }
    return FindDuplicateIndexHashed(values);
}

} // namespace rad::detail
