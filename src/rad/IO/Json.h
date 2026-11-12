#pragma once

#include <rad/Core/Result.h>
#include <rad/Core/RefCounted.h>

#include <boost/json/array.hpp>
#include <boost/json/kind.hpp>
#include <boost/json/object.hpp>
#include <boost/json/parse_options.hpp>
#include <boost/json/set_pointer_options.hpp>
#include <boost/json/string.hpp>
#include <boost/json/value.hpp>
#include <boost/system/error_code.hpp>

#include <string>
#include <string_view>

namespace rad
{

using JsonValue = boost::json::value;
using JsonObject = boost::json::object;
using JsonArray = boost::json::array;
using JsonString = boost::json::string;
using JsonKind = boost::json::kind;
using JsonErrorCode = boost::system::error_code;
using JsonParseOptions = boost::json::parse_options;
using JsonSetPointerOptions = boost::json::set_pointer_options;

// Owns a JSON root. Reference counting protects lifetime, not concurrent mutation.
// Values returned by Root, Find, and Set may be invalidated by later mutation.
// After successful schema compilation, do not modify the document, its URI, or any
// borrowed value while a compiled schema referencing it exists.
class JsonDocument final : public RefCounted<JsonDocument>
{
public:
    JsonDocument(const JsonDocument&) = delete;
    JsonDocument& operator=(const JsonDocument&) = delete;

    // Owns root; omitted root is JSON null. Lvalues are copied, rvalues are moved.
    [[nodiscard]] static Ref<JsonDocument> FromValue(JsonValue root = {});
    [[nodiscard]] static Result<Ref<JsonDocument>, JsonErrorCode> Parse(
        std::string_view text, const JsonParseOptions& options = {});

    [[nodiscard]] JsonValue& Root() noexcept;
    [[nodiscard]] const JsonValue& Root() const noexcept;
    // Optional source/retrieval URI identifying this document; initially empty.
    // Separate from JSON members such as $id. Setting it performs no validation or I/O.
    // JsonSchema uses it as the initial base before $id and the diagnostic document URI.
    [[nodiscard]] const std::string& Uri() const noexcept;
    void SetUri(std::string uri);

    // Empty JSON Pointer selects the root. Invalid or missing targets return an error.
    [[nodiscard]] Result<JsonValue*, JsonErrorCode> Find(std::string_view pointer);
    [[nodiscard]] Result<const JsonValue*, JsonErrorCode> Find(std::string_view pointer) const;
    // Inserts or replaces a value, optionally creating parents under the supplied limits.
    // On failure, already-created parents may remain (Boost.JSON's basic guarantee).
    [[nodiscard]] Result<JsonValue*, JsonErrorCode> Set(std::string_view pointer, JsonValue value,
                                                        const JsonSetPointerOptions& options = {});

private:
    explicit JsonDocument(JsonValue root);

    JsonValue m_root;
    std::string m_uri;
}; // class JsonDocument

[[nodiscard]] Result<JsonValue, JsonErrorCode> ParseJson(std::string_view text);
[[nodiscard]] Result<JsonValue, JsonErrorCode> ParseJson(std::string_view text,
                                                         const JsonParseOptions& options);

[[nodiscard]] std::string PrettyJson(const JsonValue& value, std::string_view indent = "  ");

} // namespace rad
