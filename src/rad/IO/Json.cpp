#include <rad/IO/Json.h>

#include <boost/json/parse.hpp>
#include <boost/json/serialize.hpp>
#include <boost/json/value_ref.hpp>

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

JsonDocument::JsonDocument(JsonValue root) :
    m_root(std::move(root))
{
}

Ref<JsonDocument> JsonDocument::FromValue(JsonValue root)
{
    return Ref<JsonDocument>(new JsonDocument(std::move(root)));
}

Result<Ref<JsonDocument>, JsonErrorCode> JsonDocument::Parse(std::string_view text,
                                                             const JsonParseOptions& options)
{
    auto root = ParseJson(text, options);
    if (!root)
    {
        return Failure(root.error());
    }
    return Success(FromValue(std::move(root.value())));
}

JsonValue& JsonDocument::Root() noexcept
{
    return m_root;
}

const JsonValue& JsonDocument::Root() const noexcept
{
    return m_root;
}

const std::string& JsonDocument::Uri() const noexcept
{
    return m_uri;
}

void JsonDocument::SetUri(std::string uri)
{
    m_uri = std::move(uri);
}

Result<JsonValue*, JsonErrorCode> JsonDocument::Find(std::string_view pointer)
{
    JsonErrorCode error;
    auto* value = m_root.find_pointer({pointer.data(), pointer.size()}, error);
    if (error)
    {
        return Failure(error);
    }
    return Success(value);
}

Result<const JsonValue*, JsonErrorCode> JsonDocument::Find(std::string_view pointer) const
{
    JsonErrorCode error;
    const auto* value = m_root.find_pointer({pointer.data(), pointer.size()}, error);
    if (error)
    {
        return Failure(error);
    }
    return Success(value);
}

Result<JsonValue*, JsonErrorCode> JsonDocument::Set(std::string_view pointer, JsonValue value,
                                                    const JsonSetPointerOptions& options)
{
    auto result =
        m_root.try_set_at_pointer({pointer.data(), pointer.size()}, std::move(value), options);
    if (!result)
    {
        return Failure(result.error());
    }
    return Success(&result.value());
}

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
