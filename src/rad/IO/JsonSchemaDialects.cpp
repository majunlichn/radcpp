#include "JsonSchemaDialects.h"
#include "JsonSchemaMetaSchemas.h"
#include "JsonSchemaReferences.h"

#include <boost/url.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rad::detail
{
namespace
{

[[nodiscard]] std::string_view StringView(const JsonString& text)
{
    return {text.data(), text.size()};
}

[[nodiscard]] std::optional<JsonSchemaDialect> StandardDialect(std::string_view uri)
{
    if (uri.ends_with('#'))
    {
        uri.remove_suffix(1);
    }
    if (uri == "http://json-schema.org/draft-07/schema")
    {
        return JsonSchemaDialect::Draft7;
    }
    if (uri == "https://json-schema.org/draft/2019-09/schema")
    {
        return JsonSchemaDialect::Draft2019_09;
    }
    if (uri == "https://json-schema.org/draft/2020-12/schema")
    {
        return JsonSchemaDialect::Draft2020_12;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string> NormalizeUri(std::string_view text,
                                                       bool retrieval = false)
{
    const auto parsed = boost::urls::parse_uri(text);
    if (!parsed || (parsed->has_fragment() && (retrieval || !parsed->fragment().empty())))
    {
        return std::nullopt;
    }
    boost::urls::url uri(*parsed);
    uri.normalize();
    if (!retrieval)
    {
        uri.remove_fragment();
    }
    return std::string(uri.buffer());
}

} // namespace

bool JsonSchemaVocabularyProfile::IsKeywordEnabled(std::string_view keyword) const
{
    if (dialect == JsonSchemaDialect::Draft7 || (applicator && validation && unevaluated))
    {
        return true;
    }
    constexpr std::array<std::string_view, 20> assertions = {
        "type", "enum", "const", "multipleOf", "minimum", "maximum", "exclusiveMinimum",
        "exclusiveMaximum", "minLength", "maxLength", "pattern", "minProperties",
        "maxProperties", "required", "minItems", "maxItems", "uniqueItems",
        "dependentRequired", "minContains", "maxContains",
    };
    if (std::find(assertions.begin(), assertions.end(), keyword) != assertions.end())
    {
        return validation;
    }
    constexpr std::array<std::string_view, 16> applicators = {
        "allOf", "anyOf", "oneOf", "not", "if", "then", "else", "properties",
        "patternProperties", "additionalProperties", "propertyNames", "items", "prefixItems",
        "additionalItems", "contains", "dependentSchemas",
    };
    if (std::find(applicators.begin(), applicators.end(), keyword) != applicators.end())
    {
        return applicator;
    }
    if (keyword == "unevaluatedProperties" || keyword == "unevaluatedItems")
    {
        return unevaluated;
    }
    return true;
}

class JsonSchemaDialectResolver::Impl
{
public:
    Impl(const JsonValue& root, const JsonSchemaCompileOptions& options,
         std::optional<JsonSchemaDialect> fallbackDialect, std::size_t maxDepth) :
        m_root(root),
        m_options(options),
        m_fallbackDialect(fallbackDialect),
        m_maxDepth(maxDepth)
    {
    }

    [[nodiscard]] Result<JsonSchemaVocabularyProfile, JsonSchemaCompileError>
    Resolve(std::string_view uri)
    {
        return Resolve(uri, m_options.retrievalUri, "/$schema");
    }

    [[nodiscard]] Result<JsonSchemaVocabularyProfile, JsonSchemaCompileError>
    Resolve(std::string_view uri, std::string_view sourceUri, std::string_view sourcePath)
    {
        return ResolveMeta(uri, sourceUri, sourcePath, 0);
    }

private:
    static constexpr std::size_t NoParent = std::numeric_limits<std::size_t>::max();
    enum class FailureKind
    {
        Other,
        Cycle,
    };

    struct Node
    {
        const JsonValue* value;
        std::string path;
        std::string base;
        std::string documentUri;
        std::size_t parent;
        std::string keyword;
        bool arrayItems;
        std::size_t depth;
        std::optional<JsonSchemaCompileError> identifierError;
    };

    [[nodiscard]] JsonSchemaCompileError Error(JsonSchemaCompileErrorCode code,
                                               std::string_view uri, std::string_view path,
                                               std::string message) const
    {
        m_failureKind = FailureKind::Other;
        return {code, m_fallbackDialect, std::string(path), std::move(message), std::string(uri)};
    }

    void AddIdentifier(const std::string& uri, std::size_t node)
    {
        auto& candidates = m_identifiers[uri];
        if (std::find(candidates.begin(), candidates.end(), node) == candidates.end())
        {
            candidates.push_back(node);
        }
    }

    void Scan(const JsonValue& value, std::string path, std::string base,
              const std::string& documentUri, std::size_t parent = NoParent,
              std::string keyword = {}, bool arrayItems = false, std::size_t depth = 0)
    {
        if (parent != NoParent && !value.is_object() && !value.is_bool())
        {
            return;
        }
        const auto index = m_nodes.size();
        m_nodes.push_back({&value, path, base, documentUri, parent, std::move(keyword),
                           arrayItems, depth, std::nullopt});
        if (parent == NoParent && !documentUri.empty())
        {
            AddIdentifier(documentUri, index);
        }
        if (!value.is_object())
        {
            return;
        }
        const auto& object = value.as_object();
        if (const auto* id = object.if_contains("$id"))
        {
            const auto parsed = id->is_string()
                                    ? boost::urls::parse_uri_reference(StringView(id->as_string()))
                                    : boost::urls::parse_uri_reference(" ");
            boost::urls::url resolved(base);
            if (!parsed || !resolved.resolve(*parsed) || !resolved.fragment().empty())
            {
                m_nodes[index].identifierError =
                    Error(JsonSchemaCompileErrorCode::InvalidSchema, documentUri,
                          ChildPath(path, "$id"), "$id must be a fragment-free URI reference");
            }
            else
            {
                resolved.normalize();
                resolved.remove_fragment();
                base = std::string(resolved.buffer());
                m_nodes[index].base = base;
                AddIdentifier(base, index);
            }
        }
        if (depth > m_maxDepth)
        {
            return;
        }
        // Catalogue only schema-bearing locations. Visibility is checked against each
        // ancestor's own dialect before an identifier is exported from this catalogue.
        for (std::string_view name :
             {"$defs", "definitions", "properties", "patternProperties", "dependentSchemas",
              "dependencies"})
        {
            const auto* map = object.if_contains(name);
            if (!map || !map->is_object())
            {
                continue;
            }
            for (const auto& member : map->as_object())
            {
                Scan(member.value(), ChildPath(ChildPath(path, name), member.key()), base,
                     documentUri, index, std::string(name), false, depth + 1);
            }
        }
        for (std::string_view name :
             {"additionalProperties", "propertyNames", "contains", "not", "if", "then", "else",
              "unevaluatedProperties", "unevaluatedItems", "additionalItems", "items"})
        {
            if (const auto* child = object.if_contains(name); child && !child->is_array())
            {
                Scan(*child, ChildPath(path, name), base, documentUri, index, std::string(name),
                     false, depth + 1);
            }
        }
        for (std::string_view name : {"allOf", "anyOf", "oneOf", "prefixItems", "items"})
        {
            const auto* children = object.if_contains(name);
            if (!children || !children->is_array())
            {
                continue;
            }
            for (std::size_t i = 0; i < children->as_array().size(); ++i)
            {
                Scan(children->as_array()[i], ChildPath(ChildPath(path, name), std::to_string(i)),
                     base, documentUri, index, std::string(name), name == "items", depth + 1);
            }
        }
    }

    [[nodiscard]] Result<void, JsonSchemaCompileError> Prepare()
    {
        if (m_prepared)
        {
            return m_prepareError ? Result<void, JsonSchemaCompileError>(Failure(*m_prepareError))
                                  : Result<void, JsonSchemaCompileError>(Success());
        }
        m_prepared = true;
        std::unordered_set<std::string> retrievals;
        const auto add = [&](const JsonValue& value, std::string_view text, bool root)
        {
            std::string retrieval;
            if (!text.empty() || !root)
            {
                const auto normalized = NormalizeUri(text, true);
                if (!normalized)
                {
                    m_prepareError = Error(JsonSchemaCompileErrorCode::InvalidSchema,
                                           m_options.retrievalUri, "/$schema",
                                           "meta-schema retrieval URIs must be absolute and "
                                           "fragment-free");
                    return;
                }
                retrieval = *normalized;
                if (!retrievals.insert(retrieval).second)
                {
                    m_prepareError = Error(JsonSchemaCompileErrorCode::InvalidSchema, retrieval,
                                           "", "duplicate meta-schema document retrieval URI");
                    return;
                }
            }
            Scan(value, "", retrieval.empty() ? "rad-schema://document/" : retrieval, retrieval);
        };
        add(m_root, m_options.retrievalUri, true);
        for (const auto& document : m_options.documents)
        {
            if (m_prepareError)
            {
                break;
            }
            add(document.schema, document.uri, false);
        }
        return m_prepareError ? Result<void, JsonSchemaCompileError>(Failure(*m_prepareError))
                              : Result<void, JsonSchemaCompileError>(Success());
    }

    [[nodiscard]] Result<JsonSchemaVocabularyProfile, JsonSchemaCompileError>
    Semantics(std::size_t index, std::size_t depth)
    {
        // Copy: resolving a bundled meta-schema can grow the node catalogue.
        const auto node = m_nodes[index];
        if (depth > m_maxDepth)
        {
            return Failure(Error(JsonSchemaCompileErrorCode::InvalidSchema, node.documentUri,
                                 node.path, "maximum meta-schema resolution depth exceeded"));
        }
        if (node.value->is_object())
        {
            if (const auto* schema = node.value->as_object().if_contains("$schema"))
            {
                if (!schema->is_string())
                {
                    return Failure(Error(JsonSchemaCompileErrorCode::InvalidSchema,
                                         node.documentUri, ChildPath(node.path, "$schema"),
                                         "$schema must be an absolute URI string"));
                }
                auto declared = ResolveMeta(StringView(schema->as_string()), node.documentUri,
                                            ChildPath(node.path, "$schema"), depth + 1);
                if (!declared)
                {
                    return declared;
                }
                if (node.parent != NoParent && !node.value->as_object().contains("$id"))
                {
                    const auto inherited = Semantics(node.parent, depth + 1);
                    if (!inherited)
                    {
                        return Failure(inherited.error());
                    }
                    if (declared.value().applicator != inherited.value().applicator ||
                        declared.value().validation != inherited.value().validation ||
                        declared.value().unevaluated != inherited.value().unevaluated)
                    {
                        return Failure(Error(
                            JsonSchemaCompileErrorCode::UnsupportedFeature, node.documentUri,
                            ChildPath(node.path, "$schema"),
                            "a different $schema vocabulary must be declared at a resource root"));
                    }
                }
                return declared;
            }
        }
        if (node.parent != NoParent)
        {
            return Semantics(node.parent, depth + 1);
        }
        if (m_fallbackDialect)
        {
            return Success(JsonSchemaVocabularyProfile{*m_fallbackDialect});
        }
        if (node.value != &m_root && m_root.is_object())
        {
            if (const auto* declaration = m_root.as_object().if_contains("$schema"))
            {
                if (!declaration->is_string())
                {
                    return Failure(Error(JsonSchemaCompileErrorCode::InvalidSchema,
                                         m_options.retrievalUri, "/$schema",
                                         "$schema must be an absolute URI string"));
                }
                return ResolveMeta(StringView(declaration->as_string()), m_options.retrievalUri,
                                   "/$schema", depth + 1);
            }
        }
        return Failure(Error(JsonSchemaCompileErrorCode::UnsupportedFeature, node.documentUri,
                             ChildPath(node.path, "$schema"),
                             "meta-schema has no $schema; an explicit fallback dialect "
                             "is required"));
    }

    [[nodiscard]] Result<bool, JsonSchemaCompileError> Visible(std::size_t index,
                                                              std::size_t depth)
    {
        const auto node = m_nodes[index];
        if (node.parent == NoParent)
        {
            return Success(true);
        }
        if (depth > m_maxDepth || node.depth > m_maxDepth)
        {
            return Failure(Error(JsonSchemaCompileErrorCode::InvalidSchema, node.documentUri,
                                 node.path, "maximum meta-schema resource depth exceeded"));
        }
        const auto parentVisible = Visible(node.parent, depth + 1);
        if (!parentVisible || !parentVisible.value())
        {
            return parentVisible;
        }
        if (m_nodes[node.parent].identifierError)
        {
            return Failure(*m_nodes[node.parent].identifierError);
        }
        const auto profile = Semantics(node.parent, depth + 1);
        if (!profile)
        {
            return Failure(profile.error());
        }
        const auto dialect = profile.value().dialect;
        const auto& keyword = node.keyword;
        if (!profile.value().IsKeywordEnabled(keyword) ||
            (keyword == "dependencies" && dialect != JsonSchemaDialect::Draft7) ||
            (keyword == "dependentSchemas" && dialect == JsonSchemaDialect::Draft7) ||
            (keyword == "$defs" && dialect == JsonSchemaDialect::Draft7) ||
            (keyword == "definitions" && dialect != JsonSchemaDialect::Draft7) ||
            (keyword == "prefixItems" && dialect != JsonSchemaDialect::Draft2020_12) ||
            ((keyword == "additionalItems" || node.arrayItems) &&
             dialect == JsonSchemaDialect::Draft2020_12) ||
            ((keyword == "unevaluatedItems" || keyword == "unevaluatedProperties") &&
             dialect == JsonSchemaDialect::Draft7))
        {
            return Success(false);
        }
        const auto& parent = *m_nodes[node.parent].value;
        if (dialect == JsonSchemaDialect::Draft7 && parent.is_object() &&
            parent.as_object().contains("$ref") && keyword != "definitions")
        {
            return Success(false);
        }
        if (node.value->is_object() && node.value->as_object().contains("$schema"))
        {
            const auto own = Semantics(index, depth + 1);
            if (!own)
            {
                return Failure(own.error());
            }
            if (own.value().dialect != dialect)
            {
                return Failure(Error(JsonSchemaCompileErrorCode::UnsupportedFeature,
                                     node.documentUri, ChildPath(node.path, "$schema"),
                                     "mixed underlying meta-schema drafts are not supported"));
            }
        }
        return Success(true);
    }

    [[nodiscard]] Result<std::size_t, JsonSchemaCompileError>
    Find(const std::string& uri, std::string_view sourceUri, std::string_view sourcePath,
         std::size_t depth)
    {
        m_failureKind = FailureKind::Other;
        const auto prepared = Prepare();
        if (!prepared)
        {
            return Failure(prepared.error());
        }
        std::optional<std::size_t> selected;
        // Visibility checks may append bundled nodes, invalidating map iterators.
        const auto found = m_identifiers.find(uri);
        const auto candidates = found == m_identifiers.end() ? std::vector<std::size_t>{}
                                                             : found->second;
        for (const auto candidate : candidates)
        {
            m_failureKind = FailureKind::Other;
            auto visible = Visible(candidate, depth);
            if (!visible && m_failureKind == FailureKind::Cycle)
            {
                // An embedded meta-schema can define the dialect of its containing
                // document. Determine its profile before checking ancestor visibility.
                const auto provisional = Profile(candidate, depth);
                if (!provisional)
                {
                    return Failure(provisional.error());
                }
                m_provisional.emplace(uri, provisional.value());
                m_failureKind = FailureKind::Other;
                visible = Visible(candidate, depth);
                m_provisional.erase(uri);
            }
            if (!visible)
            {
                return Failure(visible.error());
            }
            if (!visible.value())
            {
                continue;
            }
            if (selected && *selected != candidate)
            {
                const auto& node = m_nodes[candidate];
                return Failure(Error(JsonSchemaCompileErrorCode::InvalidSchema, node.documentUri,
                                     ChildPath(node.path, "$id"),
                                     "conflicting meta-schema resource identifier: " + uri));
            }
            selected = candidate;
        }
        if (selected)
        {
            const auto& node = m_nodes[*selected];
            if (node.identifierError)
            {
                return Failure(*node.identifierError);
            }
            return Success(*selected);
        }
        if (const auto text = FindJsonSchemaMetaSchema(uri))
        {
            auto parsed = ParseJson(*text);
            if (!parsed)
            {
                return Failure(Error(JsonSchemaCompileErrorCode::InvalidSchema, uri, "",
                                     "unable to parse bundled meta-schema"));
            }
            auto owned = std::make_unique<JsonValue>(std::move(parsed.value()));
            const auto index = m_nodes.size();
            Scan(*owned, "", uri, uri);
            m_builtins.push_back(std::move(owned));
            return Success(index);
        }
        return Failure(Error(JsonSchemaCompileErrorCode::UnsupportedFeature, sourceUri, sourcePath,
                             "meta-schema is not registered for offline resolution: " + uri));
    }

    [[nodiscard]] Result<JsonSchemaVocabularyProfile, JsonSchemaCompileError>
    ResolveMeta(std::string_view text, std::string_view sourceUri, std::string_view sourcePath,
                std::size_t depth)
    {
        m_failureKind = FailureKind::Other;
        const auto uri = NormalizeUri(text);
        if (!uri || (*uri != text && *uri + "#" != text))
        {
            return Failure(Error(JsonSchemaCompileErrorCode::InvalidSchema, sourceUri, sourcePath,
                                 "$schema must be a normalized absolute URI with no nonempty "
                                 "fragment"));
        }
        // Standard dialect defaults cannot be overridden by caller documents.
        if (const auto standard = StandardDialect(*uri))
        {
            return Success(JsonSchemaVocabularyProfile{*standard});
        }
        if (depth > m_maxDepth)
        {
            return Failure(Error(JsonSchemaCompileErrorCode::InvalidSchema, sourceUri, sourcePath,
                                 "maximum meta-schema resolution depth exceeded"));
        }
        if (const auto pending = m_provisional.find(*uri); pending != m_provisional.end())
        {
            return Success(pending->second);
        }
        if (m_active.contains(*uri))
        {
            auto error = Error(JsonSchemaCompileErrorCode::InvalidSchema, sourceUri, sourcePath,
                               "cyclic meta-schema $schema declaration: " + *uri);
            m_failureKind = FailureKind::Cycle;
            return Failure(std::move(error));
        }
        const auto cached = m_profiles.find(*uri);
        if (cached != m_profiles.end())
        {
            return Success(cached->second);
        }
        m_active.insert(*uri);
        const auto result = ResolveCustom(*uri, sourceUri, sourcePath, depth);
        m_active.erase(*uri);
        if (result)
        {
            m_profiles.emplace(*uri, result.value());
            m_failureKind = FailureKind::Other;
        }
        return result;
    }

    [[nodiscard]] Result<JsonSchemaVocabularyProfile, JsonSchemaCompileError>
    ResolveCustom(const std::string& uri, std::string_view sourceUri,
                  std::string_view sourcePath, std::size_t depth)
    {
        const auto selected = Find(uri, sourceUri, sourcePath, depth);
        if (!selected)
        {
            return Failure(selected.error());
        }
        return Profile(selected.value(), depth);
    }

    [[nodiscard]] Result<JsonSchemaVocabularyProfile, JsonSchemaCompileError>
    Profile(std::size_t index, std::size_t depth)
    {
        const auto node = m_nodes[index];
        if (!node.value->is_object() && !node.value->is_bool())
        {
            return Failure(Error(JsonSchemaCompileErrorCode::InvalidSchema, node.documentUri,
                                 node.path, "meta-schema must be an object or boolean"));
        }
        const auto* declaration = node.value->is_object()
                                      ? node.value->as_object().if_contains("$schema")
                                      : nullptr;
        if (!declaration && !m_fallbackDialect)
        {
            return Failure(Error(JsonSchemaCompileErrorCode::UnsupportedFeature, node.documentUri,
                                 ChildPath(node.path, "$schema"),
                                 "meta-schema has no $schema; an explicit fallback dialect "
                                 "is required"));
        }
        auto semantics = declaration ? Semantics(index, depth + 1)
                                     : Result<JsonSchemaVocabularyProfile, JsonSchemaCompileError>(
                                           Success(JsonSchemaVocabularyProfile{
                                               *m_fallbackDialect}));
        if (!semantics)
        {
            return semantics;
        }
        auto profile = JsonSchemaVocabularyProfile{semantics.value().dialect};
        if (profile.dialect == JsonSchemaDialect::Draft7)
        {
            return Failure(Error(JsonSchemaCompileErrorCode::UnsupportedFeature, node.documentUri,
                                 ChildPath(node.path, "$schema"),
                                 "custom Draft 7 meta-schema dialects are not supported"));
        }
        const auto* vocabulary = node.value->is_object()
                                     ? node.value->as_object().if_contains("$vocabulary")
                                     : nullptr;
        // These flags describe schemas selecting this meta-schema. Its own keyword
        // semantics come from its $schema, not from this $vocabulary declaration.
        // An absent $vocabulary uses standard defaults, not inherited flags or $ref.
        if (!vocabulary)
        {
            return Success(profile);
        }
        const auto path = ChildPath(node.path, "$vocabulary");
        if (!vocabulary->is_object())
        {
            return Failure(Error(JsonSchemaCompileErrorCode::InvalidSchema, node.documentUri,
                                 path, "$vocabulary must be an object"));
        }
        const auto core = JsonSchemaCoreVocabularyUri(profile.dialect);
        const auto* requiredCore = vocabulary->as_object().if_contains(core);
        if (!requiredCore || !requiredCore->is_bool() || !requiredCore->as_bool())
        {
            return Failure(Error(JsonSchemaCompileErrorCode::InvalidSchema, node.documentUri,
                                 ChildPath(path, core),
                                 "$vocabulary must require the underlying draft's "
                                 "Core vocabulary"));
        }
        profile.applicator = false;
        profile.validation = false;
        profile.unevaluated = false;
        const auto prefix = core.substr(0, core.size() - std::string_view("core").size());
        for (const auto& entry : vocabulary->as_object())
        {
            const std::string_view name = entry.key();
            const auto entryPath = ChildPath(path, name);
            if (!entry.value().is_bool())
            {
                return Failure(Error(JsonSchemaCompileErrorCode::InvalidSchema, node.documentUri,
                                     entryPath, "$vocabulary values must be booleans"));
            }
            const auto supported = GetJsonSchemaVocabularySupport(name, profile.dialect);
            if (!supported)
            {
                return Failure(Error(JsonSchemaCompileErrorCode::InvalidSchema, node.documentUri,
                                     entryPath, supported.error()));
            }
            if (!supported.value() && entry.value().as_bool())
            {
                return Failure(Error(JsonSchemaCompileErrorCode::UnsupportedFeature,
                                     node.documentUri, entryPath,
                                     "unsupported required vocabulary: " + std::string(name)));
            }
            if (supported.value() && name.starts_with(prefix))
            {
                const auto suffix = name.substr(prefix.size());
                profile.applicator |= suffix == "applicator";
                profile.validation |= suffix == "validation";
                profile.unevaluated |= suffix == "unevaluated";
            }
        }
        if (profile.dialect == JsonSchemaDialect::Draft2019_09)
        {
            profile.unevaluated = profile.applicator;
        }
        return Success(profile);
    }

    const JsonValue& m_root;
    const JsonSchemaCompileOptions& m_options;
    std::optional<JsonSchemaDialect> m_fallbackDialect;
    std::size_t m_maxDepth;
    bool m_prepared = false;
    std::optional<JsonSchemaCompileError> m_prepareError;
    mutable FailureKind m_failureKind = FailureKind::Other;
    std::vector<Node> m_nodes;
    std::unordered_map<std::string, std::vector<std::size_t>> m_identifiers;
    std::vector<std::unique_ptr<JsonValue>> m_builtins;
    std::unordered_map<std::string, JsonSchemaVocabularyProfile> m_profiles;
    std::unordered_map<std::string, JsonSchemaVocabularyProfile> m_provisional;
    std::unordered_set<std::string> m_active;
}; // class JsonSchemaDialectResolver::Impl

JsonSchemaDialectResolver::JsonSchemaDialectResolver(
    const JsonValue& root, const JsonSchemaCompileOptions& options,
    std::optional<JsonSchemaDialect> fallbackDialect, std::size_t maxDepth) :
    m_impl(std::make_unique<Impl>(root, options, fallbackDialect, maxDepth))
{
}

JsonSchemaDialectResolver::~JsonSchemaDialectResolver() = default;

Result<JsonSchemaVocabularyProfile, JsonSchemaCompileError>
JsonSchemaDialectResolver::Resolve(std::string_view uri)
{
    return m_impl->Resolve(uri);
}

Result<JsonSchemaVocabularyProfile, JsonSchemaCompileError>
JsonSchemaDialectResolver::Resolve(std::string_view uri, std::string_view sourceUri,
                                   std::string_view sourcePath)
{
    return m_impl->Resolve(uri, sourceUri, sourcePath);
}

} // namespace rad::detail
