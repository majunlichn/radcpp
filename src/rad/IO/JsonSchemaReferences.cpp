#include "JsonSchemaReferences.h"

#include <boost/url.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <optional>
#include <utility>
#include <vector>

namespace rad::detail
{
namespace
{

[[nodiscard]] std::string_view StringView(const JsonString& value)
{
    return {value.data(), value.size()};
}

[[nodiscard]] std::optional<std::string> DecodePointerToken(std::string_view token)
{
    std::string decoded;
    for (std::size_t index = 0; index < token.size(); ++index)
    {
        if (token[index] != '~')
        {
            decoded += token[index];
            continue;
        }
        if (++index == token.size() || (token[index] != '0' && token[index] != '1'))
        {
            return std::nullopt;
        }
        decoded += token[index] == '0' ? '~' : '/';
    }
    return decoded;
}

[[nodiscard]] bool IsAnchor(std::string_view value, JsonSchemaDialect dialect)
{
    const auto isLetter = [](char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); };
    if (value.empty() || (!isLetter(value.front()) &&
                          !(dialect == JsonSchemaDialect::Draft2020_12 && value.front() == '_')))
    {
        return false;
    }
    return std::all_of(value.begin() + 1, value.end(),
                       [&](char c)
                       {
                           return isLetter(c) || (c >= '0' && c <= '9') || c == '-' || c == '_' ||
                                  c == '.' ||
                                  (dialect != JsonSchemaDialect::Draft2020_12 && c == ':');
                       });
}

class ReferenceIndexBuilder
{
public:
    using Targets = std::unordered_map<std::string, JsonSchemaReferences::Resolution>;

    ReferenceIndexBuilder(const JsonValue& root, JsonSchemaDialect dialect, std::size_t maxDepth) :
        m_root(root),
        m_dialect(dialect),
        m_maxDepth(maxDepth)
    {
    }

    [[nodiscard]] Result<Targets, JsonSchemaCompileError> Build()
    {
        // A private retrieval URI gives anonymous schemas a base without enabling I/O.
        const std::string anonymousBase = "rad-schema://document/";
        // References may turn locations under otherwise unknown keywords into schemas.
        std::vector<std::string> discoveredPaths;
        for (;;)
        {
            m_resources.clear();
            m_anchors.clear();
            m_scopes.clear();
            m_references.clear();
            m_resources.emplace(anonymousBase, "");
            Index(m_root, "", anonymousBase, 0);
            // Rebuild ancestor-first so newly discovered resources replace stale scopes.
            std::sort(discoveredPaths.begin(), discoveredPaths.end());
            for (const auto& path : discoveredPaths)
            {
                Index(*FindJsonSchemaValue(m_root, path), path, ParentScope(path), 1);
            }
            if (m_error)
            {
                break;
            }
            const auto previousSize = discoveredPaths.size();
            for (const auto& source : m_references)
            {
                const auto target = Resolve(source);
                if (!target || m_scopes.contains(*target))
                {
                    continue;
                }
                if (std::find(discoveredPaths.begin(), discoveredPaths.end(), *target) ==
                    discoveredPaths.end())
                {
                    discoveredPaths.push_back(*target);
                }
            }
            if (discoveredPaths.size() == previousSize)
            {
                break;
            }
        }

        Targets targets;
        for (const auto& source : m_references)
        {
            if (m_error)
            {
                break;
            }
            const auto target = Resolve(source);
            if (target)
            {
                targets.emplace(ChildPath(source, "$ref"), Success(*target));
            }
            else if (!m_error)
            {
                targets.emplace(ChildPath(source, "$ref"), Failure(*m_resolutionError));
            }
        }
        if (m_error)
        {
            return Failure(std::move(*m_error));
        }
        return Success(std::move(targets));
    }

private:
    void Error(std::string_view path, std::string message)
    {
        if (!m_error)
        {
            m_error = JsonSchemaCompileError{JsonSchemaCompileErrorCode::InvalidSchema, m_dialect,
                                             std::string(path), std::move(message)};
        }
    }

    [[nodiscard]] std::optional<boost::urls::url> ResolveUri(std::string_view text,
                                                             std::string_view base,
                                                             std::string_view path,
                                                             bool reference = false)
    {
        const auto error = [&](std::string message)
        {
            if (reference)
            {
                m_resolutionError =
                    JsonSchemaCompileError{JsonSchemaCompileErrorCode::InvalidSchema, m_dialect,
                                           std::string(path), std::move(message)};
            }
            else
            {
                Error(path, std::move(message));
            }
        };
        const auto parsed = boost::urls::parse_uri_reference(text);
        if (!parsed)
        {
            error("invalid URI reference");
            return std::nullopt;
        }
        boost::urls::url result(base);
        const auto resolved = result.resolve(*parsed);
        if (!resolved)
        {
            error("unable to resolve URI reference: " + resolved.error().message());
            return std::nullopt;
        }
        result.normalize();
        return result;
    }

    void Register(std::unordered_map<std::string, std::string>& table, std::string uri,
                  const std::string& path, std::string_view keywordPath)
    {
        const auto [entry, inserted] = table.emplace(std::move(uri), path);
        if (!inserted && entry->second != path)
        {
            Error(keywordPath, "duplicate schema resource identifier or anchor");
        }
    }

    void Index(const JsonValue& schema, const std::string& path, std::string base,
               std::size_t depth)
    {
        if (m_error || m_scopes.contains(path))
        {
            return;
        }
        if (depth > m_maxDepth)
        {
            Error(path, "maximum schema depth exceeded");
            return;
        }
        if (!schema.is_object())
        {
            m_scopes.emplace(path, std::move(base));
            return;
        }
        const auto& object = schema.as_object();
        const bool ignoredSiblings =
            m_dialect == JsonSchemaDialect::Draft7 && object.contains("$ref");
        if (!ignoredSiblings)
        {
            if (const auto* id = object.if_contains("$id"))
            {
                const auto keywordPath = ChildPath(path, "$id");
                if (!id->is_string())
                {
                    Error(keywordPath, "$id must be a URI-reference string");
                    return;
                }
                auto uri = ResolveUri(StringView(id->as_string()), base, keywordPath);
                if (!uri)
                {
                    return;
                }
                const auto fragment = uri->fragment();
                if (!fragment.empty() &&
                    (m_dialect != JsonSchemaDialect::Draft7 || !IsAnchor(fragment, m_dialect)))
                {
                    Error(keywordPath, "invalid $id fragment for this dialect");
                    return;
                }
                auto previous = boost::urls::url(base);
                previous.remove_fragment();
                base = std::string(uri->buffer());
                if (!fragment.empty())
                {
                    Register(m_anchors, base, path, keywordPath);
                }
                uri->remove_fragment();
                if (fragment.empty() || uri->buffer() != previous.buffer())
                {
                    Register(m_resources, std::string(uri->buffer()), path, keywordPath);
                }
            }
            if (m_dialect != JsonSchemaDialect::Draft7)
            {
                if (const auto* anchor = object.if_contains("$anchor"))
                {
                    const auto keywordPath = ChildPath(path, "$anchor");
                    if (!anchor->is_string() ||
                        !IsAnchor(StringView(anchor->as_string()), m_dialect))
                    {
                        Error(keywordPath, "$anchor must be a valid plain-name identifier");
                        return;
                    }
                    auto uri = boost::urls::url(base);
                    uri.set_fragment(StringView(anchor->as_string()));
                    uri.normalize();
                    Register(m_anchors, std::string(uri.buffer()), path, keywordPath);
                }
            }
        }
        m_scopes.emplace(path, base);
        if (object.contains("$ref"))
        {
            m_references.push_back(path);
        }

        const auto indexMap = [&](std::string_view keyword, bool dependencies = false)
        {
            const auto* value = object.if_contains(keyword);
            if (value == nullptr || !value->is_object())
            {
                return;
            }
            const auto keywordPath = ChildPath(path, keyword);
            for (const auto& member : value->as_object())
            {
                if (!dependencies || !member.value().is_array())
                {
                    Index(member.value(), ChildPath(keywordPath, member.key()), base, depth + 1);
                }
            }
        };
        indexMap(m_dialect == JsonSchemaDialect::Draft7 ? "definitions" : "$defs");
        if (ignoredSiblings)
        {
            return;
        }
        indexMap("properties");
        indexMap("patternProperties");
        if (m_dialect == JsonSchemaDialect::Draft7)
        {
            indexMap("dependencies", true);
        }
        else
        {
            indexMap("dependentSchemas");
        }

        constexpr std::array singles = {
            "additionalProperties", "propertyNames", "contains", "not", "if", "then", "else",
        };
        for (const std::string_view keyword : singles)
        {
            if (const auto* value = object.if_contains(keyword))
            {
                Index(*value, ChildPath(path, keyword), base, depth + 1);
            }
        }
        if (m_dialect != JsonSchemaDialect::Draft7)
        {
            for (const std::string_view keyword : {"unevaluatedItems", "unevaluatedProperties"})
            {
                if (const auto* value = object.if_contains(keyword))
                {
                    Index(*value, ChildPath(path, keyword), base, depth + 1);
                }
            }
        }
        if (m_dialect != JsonSchemaDialect::Draft2020_12)
        {
            if (const auto* value = object.if_contains("additionalItems"))
            {
                Index(*value, ChildPath(path, "additionalItems"), base, depth + 1);
            }
        }
        const auto indexArray = [&](const JsonArray& values, std::string_view keyword)
        {
            const auto keywordPath = ChildPath(path, keyword);
            for (std::size_t index = 0; index < values.size(); ++index)
            {
                Index(values[index], ChildPath(keywordPath, std::to_string(index)), base,
                      depth + 1);
            }
        };
        for (const std::string_view keyword : {"allOf", "anyOf", "oneOf"})
        {
            if (const auto* value = object.if_contains(keyword); value && value->is_array())
            {
                indexArray(value->as_array(), keyword);
            }
        }
        if (m_dialect == JsonSchemaDialect::Draft2020_12)
        {
            if (const auto* value = object.if_contains("prefixItems"); value && value->is_array())
            {
                indexArray(value->as_array(), "prefixItems");
            }
        }
        if (const auto* items = object.if_contains("items"))
        {
            if (items->is_array() && m_dialect != JsonSchemaDialect::Draft2020_12)
            {
                indexArray(items->as_array(), "items");
            }
            else
            {
                Index(*items, ChildPath(path, "items"), base, depth + 1);
            }
        }
    }

    [[nodiscard]] std::string ParentScope(std::string path) const
    {
        for (;;)
        {
            const auto end = path.rfind('/');
            path = end == std::string::npos ? "" : path.substr(0, end);
            if (const auto found = m_scopes.find(path); found != m_scopes.end())
            {
                return found->second;
            }
        }
    }

    [[nodiscard]] std::optional<std::string> Resolve(const std::string& source)
    {
        m_resolutionError.reset();
        const auto keywordPath = ChildPath(source, "$ref");
        const auto* schema = FindJsonSchemaValue(m_root, source);
        const auto& reference = schema->as_object().at("$ref");
        if (!reference.is_string())
        {
            m_resolutionError =
                JsonSchemaCompileError{JsonSchemaCompileErrorCode::InvalidSchema, m_dialect,
                                       keywordPath, "$ref must be a string"};
            return std::nullopt;
        }
        auto uri =
            ResolveUri(StringView(reference.as_string()), m_scopes.at(source), keywordPath, true);
        if (!uri)
        {
            return std::nullopt;
        }
        const auto fragment = uri->fragment();
        const auto anchor = m_anchors.find(std::string(uri->buffer()));
        if (anchor != m_anchors.end())
        {
            return anchor->second;
        }
        uri->remove_fragment();
        const auto resource = m_resources.find(std::string(uri->buffer()));
        if (resource == m_resources.end())
        {
            m_resolutionError = JsonSchemaCompileError{
                JsonSchemaCompileErrorCode::UnsupportedFeature, m_dialect, keywordPath,
                "references to external schema resources are not supported"};
            return std::nullopt;
        }
        if (!fragment.empty() && fragment.front() != '/')
        {
            m_resolutionError =
                JsonSchemaCompileError{JsonSchemaCompileErrorCode::InvalidSchema, m_dialect,
                                       keywordPath, "$ref anchor does not exist"};
            return std::nullopt;
        }
        const auto targetPath = resource->second + fragment;
        if (FindJsonSchemaValue(m_root, targetPath) == nullptr)
        {
            m_resolutionError = JsonSchemaCompileError{
                JsonSchemaCompileErrorCode::InvalidSchema, m_dialect, keywordPath,
                "$ref target does not exist or has an invalid JSON Pointer"};
            return std::nullopt;
        }
        return targetPath;
    }

    const JsonValue& m_root;
    JsonSchemaDialect m_dialect;
    std::size_t m_maxDepth;
    std::unordered_map<std::string, std::string> m_resources;
    std::unordered_map<std::string, std::string> m_anchors;
    std::unordered_map<std::string, std::string> m_scopes;
    std::vector<std::string> m_references;
    std::optional<JsonSchemaCompileError> m_error;
    std::optional<JsonSchemaCompileError> m_resolutionError;
}; // class ReferenceIndexBuilder

} // namespace

std::string ChildPath(std::string_view path, std::string_view token)
{
    std::string child(path);
    child += '/';
    for (const char c : token)
    {
        if (c == '~')
        {
            child += "~0";
        }
        else if (c == '/')
        {
            child += "~1";
        }
        else
        {
            child += c;
        }
    }
    return child;
}

const JsonValue* FindJsonSchemaValue(const JsonValue& root, std::string_view pointer)
{
    const JsonValue* value = &root;
    if (!pointer.empty() && pointer.front() != '/')
    {
        return nullptr;
    }
    for (std::size_t begin = 1; begin <= pointer.size();)
    {
        const auto end = pointer.find('/', begin);
        const auto token = DecodePointerToken(
            pointer.substr(begin, end == std::string_view::npos ? end : end - begin));
        if (!token)
        {
            return nullptr;
        }
        if (value->is_object())
        {
            value = value->as_object().if_contains(*token);
        }
        else if (value->is_array())
        {
            std::size_t index = 0;
            const auto [next, error] =
                std::from_chars(token->data(), token->data() + token->size(), index);
            value = !token->empty() && (token->size() == 1 || token->front() != '0') &&
                            error == std::errc{} && next == token->data() + token->size() &&
                            index < value->as_array().size()
                        ? &value->as_array()[index]
                        : nullptr;
        }
        else
        {
            return nullptr;
        }
        if (value == nullptr)
        {
            return nullptr;
        }
        if (end == std::string_view::npos)
        {
            break;
        }
        begin = end + 1;
    }
    return value;
}

Result<JsonSchemaReferences, JsonSchemaCompileError> JsonSchemaReferences::Compile(
    const JsonValue& schema, JsonSchemaDialect dialect, std::size_t maxDepth)
{
    ReferenceIndexBuilder builder(schema, dialect, maxDepth);
    auto targets = builder.Build();
    if (!targets)
    {
        return Failure(std::move(targets.error()));
    }
    JsonSchemaReferences references;
    references.m_targets = std::move(targets.value());
    return Success(std::move(references));
}

const JsonSchemaReferences::Resolution* JsonSchemaReferences::Find(
    std::string_view referencePath) const
{
    const auto found = m_targets.find(std::string(referencePath));
    return found == m_targets.end() ? nullptr : &found->second;
}

} // namespace rad::detail
