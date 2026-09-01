#include <rad/IO/Json.h>

#include <boost/json/parse.hpp>
#include <boost/json/serialize.hpp>

#include <cstddef>
#include <utility>

namespace rad
{
namespace
{

void PrettyJsonImpl(const JsonValue& value, std::string_view indent, std::string& output,
                    std::string& currentIndent)
{
    switch (value.kind())
    {
    case JsonKind::array:
    {
        const auto& array = value.as_array();
        if (array.empty())
        {
            output += "[]";
            return;
        }

        output += "[\n";
        currentIndent += indent;
        for (std::size_t i = 0; i < array.size(); ++i)
        {
            output += currentIndent;
            PrettyJsonImpl(array[i], indent, output, currentIndent);
            if (i + 1 != array.size())
            {
                output += ',';
            }
            output += '\n';
        }
        currentIndent.resize(currentIndent.size() - indent.size());
        output += currentIndent;
        output += ']';
        return;
    }

    case JsonKind::object:
    {
        const auto& object = value.as_object();
        if (object.empty())
        {
            output += "{}";
            return;
        }

        output += "{\n";
        currentIndent += indent;
        std::size_t i = 0;
        for (const auto& member : object)
        {
            output += currentIndent;
            output += boost::json::serialize(member.key());
            output += ": ";
            PrettyJsonImpl(member.value(), indent, output, currentIndent);
            if (++i != object.size())
            {
                output += ',';
            }
            output += '\n';
        }
        currentIndent.resize(currentIndent.size() - indent.size());
        output += currentIndent;
        output += '}';
        return;
    }

    default:
        output += boost::json::serialize(value);
        return;
    }
}

} // namespace

Result<JsonValue, JsonErrorCode> ParseJson(std::string_view text)
{
    JsonParseOptions options = {};
    return ParseJson(text, options);
}

Result<JsonValue, JsonErrorCode> ParseJson(std::string_view text,
                                          const JsonParseOptions& options)
{
    JsonErrorCode error;
    auto value = boost::json::parse({text.data(), text.size()}, error, {}, options);
    if (error)
    {
        return Failure(error);
    }
    return Success(std::move(value));
}

std::string PrettyJson(const JsonValue& value, std::string_view indent)
{
    std::string output;
    std::string currentIndent;
    PrettyJsonImpl(value, indent, output, currentIndent);
    return output;
}

} // namespace rad
