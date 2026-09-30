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

    ReferenceIndexBuilder(JsonValue& root, std::vector<std::string>& uris,
                          std::unordered_map<std::string, std::string>& schemaResources,
                          const JsonSchemaCompileOptions& options, JsonSchemaDialect dialect,
                          std::size_t maxDepth) :
        m_root(root),
        m_uris(uris),
        m_schemaResources(schemaResources),
        m_options(options),
        m_dialect(dialect),
        m_maxDepth(maxDepth)
    {
    }

    [[nodiscard]] Result<Targets, JsonSchemaCompileError> Build()
    {
        if (!PrepareRegistry())
        {
            return Failure(std::move(*m_error));
        }
        // References may turn locations under otherwise unknown keywords into schemas.
        std::vector<std::string> discoveredPaths;
        for (;;)
        {
            m_resources.clear();
            m_anchors.clear();
            m_scopes.clear();
            m_references.clear();
            m_documentErrors.clear();
            for (std::size_t index = 0; index < m_uris.size(); ++index)
            {
                const auto path = ChildPath("", std::to_string(index));
                const auto base = m_uris[index].empty() ? "rad-schema://document/" : m_uris[index];
                Register(m_resources, base, path, path);
                Index(m_root.as_array()[index], path, base, 0);
                DeferDocumentError();
            }
            if (m_error)
            {
                break;
            }
            // Rebuild ancestor-first so newly discovered resources replace stale scopes.
            std::sort(discoveredPaths.begin(), discoveredPaths.end());
            for (const auto& path : discoveredPaths)
            {
                Index(*FindJsonSchemaValue(m_root, path), path, ParentScope(path), 1);
                DeferDocumentError();
            }
            if (m_error)
            {
                break;
            }
            const auto previousSize = discoveredPaths.size();
            const auto previousDocuments = m_uris.size();
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
            if (discoveredPaths.size() == previousSize && m_uris.size() == previousDocuments)
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
                targets.emplace(ChildPath(source.path, source.keyword), Success(*target));
            }
            else if (!m_error)
            {
                targets.emplace(ChildPath(source.path, source.keyword), Failure(*m_resolutionError));
            }
        }
        if (m_error)
        {
            return Failure(std::move(*m_error));
        }
        if (m_dialect == JsonSchemaDialect::Draft2019_09)
        {
            for (const auto& [path, base] : m_scopes)
            {
                auto uri = boost::urls::url(base);
                uri.remove_fragment();
                if (const auto resource = m_resources.find(std::string(uri.buffer()));
                    resource != m_resources.end())
                {
                    m_schemaResources.emplace(path, resource->second);
                }
            }
        }
        return Success(std::move(targets));
    }

private:
    [[nodiscard]] static std::string DocumentPath(std::string_view path)
    {
        return std::string(path.substr(0, path.find('/', 1)));
    }

    void DeferDocumentError()
    {
        if (m_error && DocumentPath(m_error->schemaPath) != "/0")
        {
            const auto path = DocumentPath(m_error->schemaPath);
            m_documentErrors.emplace(path, std::move(*m_error));
            m_error.reset();
        }
    }

    struct RegistryEntry
    {
        const JsonSchemaDocument* document;
        std::string retrievalUri;
    };

    struct PendingReference
    {
        std::string path;
        std::string_view keyword;
    };

    [[nodiscard]] std::optional<std::string> DocumentUri(std::string_view text)
    {
        const auto parsed = boost::urls::parse_uri(text);
        if (!parsed || parsed->has_fragment())
        {
            Error("/0", "retrieval URIs must be absolute and fragment-free");
            return std::nullopt;
        }
        boost::urls::url uri(*parsed);
        uri.normalize();
        return std::string(uri.buffer());
    }

    [[nodiscard]] bool PrepareRegistry()
    {
        if (!m_options.retrievalUri.empty())
        {
            const auto uri = DocumentUri(m_options.retrievalUri);
            if (!uri)
            {
                return false;
            }
            m_uris[0] = *uri;
        }
        for (const auto& document : m_options.documents)
        {
            const auto uri = DocumentUri(document.uri);
            if (!uri)
            {
                return false;
            }
            if (*uri == m_uris[0] ||
                !m_registry.emplace(*uri, RegistryEntry{&document, *uri}).second)
            {
                Error("/0", "duplicate retrieval URI in schema document registry");
                return false;
            }
        }
        for (const auto& document : m_options.documents)
        {
            auto retrieval = boost::urls::url(document.uri);
            retrieval.normalize();
            m_resources.clear();
            m_anchors.clear();
            m_scopes.clear();
            m_references.clear();
            m_cataloging = true;
            Index(document.schema, "/1", std::string(retrieval.buffer()), 0);
            m_cataloging = false;
            // Identifier errors are reported if this document becomes reachable.
            m_error.reset();
            for (const auto& [uri, path] : m_resources)
            {
                const auto [found, inserted] = m_registry.emplace(
                    uri, RegistryEntry{&document, std::string(retrieval.buffer())});
                if (!inserted && found->second.document != &document)
                {
                    Error("/0", "conflicting document identifiers in schema registry");
                    return false;
                }
            }
        }
        return true;
    }

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
        if (!m_cataloging && &table == &m_resources)
        {
            const auto registered = m_registry.find(uri);
            if (registered != m_registry.end())
            {
                const auto end = path.find('/', 1);
                const auto token =
                    std::string_view(path).substr(1, end == std::string::npos ? end : end - 1);
                std::size_t index = 0;
                const auto [next, error] =
                    std::from_chars(token.data(), token.data() + token.size(), index);
                if (error == std::errc{} && next == token.data() + token.size() &&
                    index < m_uris.size() && m_uris[index] != registered->second.retrievalUri)
                {
                    Error(keywordPath, "schema resource conflicts with a registered document");
                    return;
                }
            }
        }
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
            if (path.find('/', 1) == std::string::npos && !schema.is_bool())
            {
                Error(path, "schema must be an object or boolean");
                return;
            }
            m_scopes.emplace(path, std::move(base));
            return;
        }
        const auto& object = schema.as_object();
        if (!m_cataloging && path.find('/', 1) == std::string::npos &&
            !(path == "/0" && m_dialect == JsonSchemaDialect::Draft7 && object.contains("$ref")))
        {
            if (const auto* declared = object.if_contains("$schema"); declared != nullptr)
            {
                const std::string_view expected =
                    m_dialect == JsonSchemaDialect::Draft7
                        ? "http://json-schema.org/draft-07/schema"
                    : m_dialect == JsonSchemaDialect::Draft2019_09
                        ? "https://json-schema.org/draft/2019-09/schema"
                        : "https://json-schema.org/draft/2020-12/schema";
                auto text = declared->is_string() ? StringView(declared->as_string()) : "";
                if (text.ends_with('#'))
                {
                    text.remove_suffix(1);
                }
                if (text != expected)
                {
                    m_error = JsonSchemaCompileError{
                        declared->is_string() ? JsonSchemaCompileErrorCode::UnsupportedFeature
                                              : JsonSchemaCompileErrorCode::InvalidSchema,
                        m_dialect, ChildPath(path, "$schema"),
                        declared->is_string()
                            ? "custom or mismatched meta-schemas are not supported"
                            : "$schema must be a string"};
                    return;
                }
            }
        }
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
            m_references.push_back({path, "$ref"});
        }
        if (m_dialect == JsonSchemaDialect::Draft2019_09 && object.contains("$recursiveRef"))
        {
            m_references.push_back({path, "$recursiveRef"});
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

    [[nodiscard]] std::optional<std::string> Resolve(const PendingReference& source)
    {
        m_resolutionError.reset();
        if (const auto failed = m_documentErrors.find(DocumentPath(source.path));
            failed != m_documentErrors.end())
        {
            m_resolutionError = failed->second;
            return std::nullopt;
        }
        const auto keywordPath = ChildPath(source.path, source.keyword);
        const auto* schema = FindJsonSchemaValue(m_root, source.path);
        const auto& reference = schema->as_object().at(source.keyword);
        if (!reference.is_string())
        {
            m_resolutionError =
                JsonSchemaCompileError{JsonSchemaCompileErrorCode::InvalidSchema, m_dialect,
                                       keywordPath, std::string(source.keyword) + " must be a string"};
            return std::nullopt;
        }
        auto uri =
            ResolveUri(StringView(reference.as_string()), m_scopes.at(source.path),
                       keywordPath, true);
        if (!uri)
        {
            return std::nullopt;
        }
        if (source.keyword == "$recursiveRef" && StringView(reference.as_string()) != "#")
        {
            m_resolutionError = JsonSchemaCompileError{
                JsonSchemaCompileErrorCode::UnsupportedFeature, m_dialect, keywordPath,
                "$recursiveRef supports only the value \"#\""};
            return std::nullopt;
        }
        const auto fragment = uri->fragment();
        const auto anchor = m_anchors.find(std::string(uri->buffer()));
        if (anchor != m_anchors.end())
        {
            if (const auto failed = m_documentErrors.find(DocumentPath(anchor->second));
                failed != m_documentErrors.end())
            {
                m_resolutionError = failed->second;
                return std::nullopt;
            }
            return anchor->second;
        }
        uri->remove_fragment();
        const auto resource = m_resources.find(std::string(uri->buffer()));
        if (resource == m_resources.end())
        {
            const auto document = m_registry.find(std::string(uri->buffer()));
            if (document != m_registry.end())
            {
                // Loading only copies caller-supplied values; resolution never performs I/O.
                const auto& entry = document->second;
                if (std::find(m_uris.begin(), m_uris.end(), entry.retrievalUri) == m_uris.end())
                {
                    m_root.as_array().push_back(entry.document->schema);
                    m_uris.push_back(entry.retrievalUri);
                }
            }
            m_resolutionError = JsonSchemaCompileError{
                JsonSchemaCompileErrorCode::UnsupportedFeature, m_dialect, keywordPath,
                "schema document is not registered: " + std::string(uri->buffer())};
            return std::nullopt;
        }
        if (const auto failed = m_documentErrors.find(DocumentPath(resource->second));
            failed != m_documentErrors.end())
        {
            m_resolutionError = failed->second;
            return std::nullopt;
        }
        if (!fragment.empty() && fragment.front() != '/')
        {
            auto canonical = boost::urls::url(m_scopes.at(resource->second));
            canonical.set_fragment(fragment);
            canonical.normalize();
            const auto named = m_anchors.find(std::string(canonical.buffer()));
            if (named != m_anchors.end())
            {
                return named->second;
            }
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

    JsonValue& m_root;
    std::vector<std::string>& m_uris;
    std::unordered_map<std::string, std::string>& m_schemaResources;
    const JsonSchemaCompileOptions& m_options;
    std::unordered_map<std::string, RegistryEntry> m_registry;
    std::unordered_map<std::string, JsonSchemaCompileError> m_documentErrors;
    JsonSchemaDialect m_dialect;
    std::size_t m_maxDepth;
    std::unordered_map<std::string, std::string> m_resources;
    std::unordered_map<std::string, std::string> m_anchors;
    std::unordered_map<std::string, std::string> m_scopes;
    std::vector<PendingReference> m_references;
    std::optional<JsonSchemaCompileError> m_error;
    std::optional<JsonSchemaCompileError> m_resolutionError;
    bool m_cataloging = false;
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
    const JsonValue& schema, JsonSchemaDialect dialect, std::size_t maxDepth,
    const JsonSchemaCompileOptions& options)
{
    JsonSchemaReferences references;
    references.m_documents = JsonArray{schema};
    references.m_uris.emplace_back();
    ReferenceIndexBuilder builder(references.m_documents, references.m_uris,
                                  references.m_schemaResources, options, dialect, maxDepth);
    auto targets = builder.Build();
    if (!targets)
    {
        auto error = std::move(targets.error());
        error.schemaUri = references.SchemaUri(error.schemaPath);
        error.schemaPath = references.SchemaPath(error.schemaPath);
        return Failure(std::move(error));
    }
    references.m_targets = std::move(targets.value());
    return Success(std::move(references));
}

const JsonValue& JsonSchemaReferences::Documents() const
{
    return m_documents;
}

std::string JsonSchemaReferences::SchemaPath(std::string_view path) const
{
    const auto end = path.find('/', 1);
    return end == std::string_view::npos ? "" : std::string(path.substr(end));
}

std::string JsonSchemaReferences::SchemaUri(std::string_view path) const
{
    const auto end = path.find('/', 1);
    const auto token = path.substr(1, end == std::string_view::npos ? end : end - 1);
    std::size_t index = 0;
    const auto [next, error] = std::from_chars(token.data(), token.data() + token.size(), index);
    return error == std::errc{} && next == token.data() + token.size() && index < m_uris.size()
               ? m_uris[index]
               : "";
}

const JsonSchemaReferences::Resolution* JsonSchemaReferences::Find(
    std::string_view referencePath) const
{
    const auto found = m_targets.find(std::string(referencePath));
    return found == m_targets.end() ? nullptr : &found->second;
}

const std::string* JsonSchemaReferences::Resource(std::string_view schemaPath) const
{
    const auto found = m_schemaResources.find(std::string(schemaPath));
    return found == m_schemaResources.end() ? nullptr : &found->second;
}

bool JsonSchemaReferences::HasRecursiveAnchor(std::string_view resourcePath) const
{
    const auto* schema = FindJsonSchemaValue(m_documents, resourcePath);
    if (schema == nullptr || !schema->is_object())
    {
        return false;
    }
    const auto* anchor = schema->as_object().if_contains("$recursiveAnchor");
    return anchor != nullptr && anchor->is_bool() && anchor->as_bool();
}

} // namespace rad::detail
