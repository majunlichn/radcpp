#include "JsonSchemaDialects.h"
#include "JsonSchemaMetaSchemas.h"
#include "JsonSchemaKeywords.h"
#include "JsonSchemaVocabularies.h"

#include <boost/url.hpp>

#include <algorithm>
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
    const auto* description = FindJsonSchemaKeyword(keyword);
    if (description == nullptr || description->Shape(dialect) == JsonSchemaChildShape::Unsupported)
    {
        return true;
    }
    switch (description->vocabulary)
    {
    case JsonSchemaKeywordVocabulary::Validation:
        return validation;
    case JsonSchemaKeywordVocabulary::Applicator:
        return applicator;
    case JsonSchemaKeywordVocabulary::Unevaluated:
        return unevaluated;
    case JsonSchemaKeywordVocabulary::Core:
        return true;
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
        bool arrayChild;
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
              std::string keyword = {}, bool arrayChild = false, std::size_t depth = 0)
    {
        if (parent != NoParent && !value.is_object() && !value.is_bool())
        {
            return;
        }
        const auto index = m_nodes.size();
        m_nodes.push_back({&value, path, base, documentUri, parent, std::move(keyword), arrayChild,
                           depth, std::nullopt});
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
        for (const auto& description : JsonSchemaChildKeywords())
        {
            const auto* child = object.if_contains(description.name);
            if (child == nullptr)
            {
                continue;
            }
            ForEachJsonSchemaChild(
                *child, ChildPath(path, description.name), description.CandidateShape(),
                [&](const JsonValue& schema, std::string childPath, bool arrayChild)
                {
                    Scan(schema, std::move(childPath), base, documentUri, index,
                         std::string(description.name), arrayChild, depth + 1);
                });
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
        const auto* description = FindJsonSchemaKeyword(keyword);
        if (description == nullptr || !profile.value().IsKeywordEnabled(keyword) ||
            !description->AcceptsChild(dialect, node.arrayChild))
        {
            return Success(false);
        }
        const auto& parent = *m_nodes[node.parent].value;
        if (dialect == JsonSchemaDialect::Draft7 && parent.is_object() &&
            parent.as_object().contains("$ref") && !description->retainedBesideDraft7Ref)
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
        const auto parsed = ParseJsonSchemaVocabularies(*vocabulary, profile.dialect);
        if (!parsed)
        {
            const auto& error = parsed.error();
            auto path = ChildPath(node.path, "$vocabulary") + error.path;
            auto message = error.message;
            auto code = JsonSchemaCompileErrorCode::InvalidSchema;
            switch (error.kind)
            {
            case JsonSchemaVocabularyErrorKind::MissingCore:
                path = ChildPath(path, JsonSchemaCoreVocabularyUri(profile.dialect));
                [[fallthrough]];
            case JsonSchemaVocabularyErrorKind::InvalidCore:
                message = "$vocabulary must require the underlying draft's Core vocabulary";
                break;
            case JsonSchemaVocabularyErrorKind::InvalidRequirement:
                message = "$vocabulary values must be booleans";
                break;
            case JsonSchemaVocabularyErrorKind::UnsupportedRequired:
                code = JsonSchemaCompileErrorCode::UnsupportedFeature;
                message = "unsupported required vocabulary: " + message;
                break;
            default:
                break;
            }
            return Failure(Error(code, node.documentUri, path, std::move(message)));
        }
        profile.applicator = parsed.value().applicator;
        profile.validation = parsed.value().validation;
        profile.unevaluated = parsed.value().unevaluated;
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
