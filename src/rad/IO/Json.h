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
#include <type_traits>

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

// Non-owning, non-null view of an existing JSON value; copies only the pointer.
// The value must outlive the view. Container mutation may invalidate child views.
// Constness of the view does not restrict mutation; constness of Value does.
template <typename Value>
class JsonValueRef
{
    static_assert(std::is_same_v<Value, JsonValue> || std::is_same_v<Value, const JsonValue>);

public:
    explicit JsonValueRef(Value& value) noexcept :
        m_value(&value)
    {
    }
    JsonValueRef(JsonValue&&) = delete;
    JsonValueRef(const JsonValue&&) = delete;

    template <typename Other>
        requires(std::is_const_v<Value> && std::is_same_v<Other, JsonValue>)
    JsonValueRef(const JsonValueRef<Other>& other) noexcept :
        m_value(&other.Get())
    {
    }

    [[nodiscard]] Value& Get() const noexcept { return *m_value; }
    [[nodiscard]] Value* operator->() const noexcept { return m_value; }

    [[nodiscard]] JsonKind Kind() const noexcept { return m_value->kind(); }
    [[nodiscard]] bool IsObject() const noexcept { return m_value->is_object(); }
    [[nodiscard]] bool IsArray() const noexcept { return m_value->is_array(); }
    [[nodiscard]] bool IsString() const noexcept { return m_value->is_string(); }
    [[nodiscard]] bool IsInt64() const noexcept { return m_value->is_int64(); }
    [[nodiscard]] bool IsUInt64() const noexcept { return m_value->is_uint64(); }
    [[nodiscard]] bool IsDouble() const noexcept { return m_value->is_double(); }
    [[nodiscard]] bool IsNumber() const noexcept { return m_value->is_number(); }
    [[nodiscard]] bool IsBool() const noexcept { return m_value->is_bool(); }
    [[nodiscard]] bool IsNull() const noexcept { return m_value->is_null(); }

    // As* preserves Boost.JSON's reference/value return types and throws on a wrong kind.
    [[nodiscard]] decltype(auto) AsObject() const { return m_value->as_object(); }
    [[nodiscard]] decltype(auto) AsArray() const { return m_value->as_array(); }
    [[nodiscard]] decltype(auto) AsString() const { return m_value->as_string(); }
    [[nodiscard]] decltype(auto) AsInt64() const { return m_value->as_int64(); }
    [[nodiscard]] decltype(auto) AsUInt64() const { return m_value->as_uint64(); }
    [[nodiscard]] decltype(auto) AsDouble() const { return m_value->as_double(); }
    [[nodiscard]] decltype(auto) AsBool() const { return m_value->as_bool(); }

    // JSON Pointer lookup; empty selects this value. Errors are Boost.JSON error codes.
    [[nodiscard]] Result<JsonValueRef, JsonErrorCode> Find(std::string_view pointer) const;
    // Direct object key or array index lookup, without insertion or JSON Pointer decoding.
    // Missing elements or wrong container kinds return Boost.JSON errors.
    [[nodiscard]] Result<JsonValueRef, JsonErrorCode> TryAt(std::string_view key) const;
    [[nodiscard]] Result<JsonValueRef, JsonErrorCode> TryAt(std::size_t index) const;

private:
    Value* m_value;
}; // class JsonValueRef

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
