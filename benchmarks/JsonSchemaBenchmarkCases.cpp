#include "JsonSchemaBenchmarkCases.h"

#include <stdexcept>
#include <string_view>
#include <utility>
#include <cstdint>
#include <format>

namespace benchmark
{
namespace
{

using rad::JsonArray;
using rad::JsonObject;
using rad::JsonValue;

[[nodiscard]] JsonValue Parse(std::string_view text)
{
    auto result = rad::ParseJson(text);
    if (!result)
    {
        throw std::runtime_error("Invalid benchmark JSON: " + result.error().message());
    }
    return std::move(result).value();
}

[[nodiscard]] JsonSchemaBenchmarkCase MakeCase(std::string name, std::string_view schemaText)
{
    auto schema = Parse(schemaText);
    schema.as_object()["$schema"] = "https://json-schema.org/draft/2020-12/schema";
    return {std::move(name),
            rad::JsonDocument::FromValue(std::move(schema)),
            {{"valid", {}, 0}, {"invalid", {}, 1}}};
}

void AddPair(JsonSchemaBenchmarkCase& testCase, JsonValue valid, JsonValue invalid)
{
    testCase.corpora[0].instances.push_back(std::move(valid));
    testCase.corpora[1].instances.push_back(std::move(invalid));
}

[[nodiscard]] JsonSchemaBenchmarkCase MakeServiceConfig()
{
    auto testCase = MakeCase("service_config", R"json({
        "type": "object",
        "required": ["name", "port", "workers", "logging", "features"],
        "additionalProperties": false,
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 64},
            "port": {"type": "integer", "minimum": 1, "maximum": 65535},
            "workers": {"type": "integer", "minimum": 1, "maximum": 256},
            "logging": {
                "type": "object", "required": ["level", "file"],
                "additionalProperties": false,
                "properties": {
                    "level": {"enum": ["debug", "info", "warn", "error"]},
                    "file": {"type": "string", "pattern": "^[a-z0-9_-]+\\.log$"}
                }
            },
            "features": {
                "type": "array", "maxItems": 16, "uniqueItems": true,
                "items": {"enum": ["cache", "compression", "metrics", "tracing"]}
            }
        }
    })json");
    testCase.corpora[1].expectedErrors = 3;
    for (std::int64_t index = 0; index < 16; ++index)
    {
        auto valid = Parse(R"json({
            "name": "gateway", "port": 8080, "workers": 8,
            "logging": {"level": "info", "file": "gateway.log"},
            "features": ["cache", "compression", "metrics"]
        })json");
        valid.as_object()["name"] = "gateway-" + std::to_string(index);
        valid.as_object()["port"] = 8080 + index;
        auto invalid = valid;
        invalid.as_object()["port"] = 0;
        invalid.as_object()["workers"] = 0;
        invalid.as_object().at("logging").as_object()["level"] = "verbose";
        AddPair(testCase, std::move(valid), std::move(invalid));
    }
    return testCase;
}

[[nodiscard]] JsonSchemaBenchmarkCase MakeOrder()
{
    auto testCase = MakeCase("nested_order", R"json({
        "type": "object",
        "required": ["id", "customer", "items", "delivery"],
        "additionalProperties": false,
        "properties": {
            "id": {"type": "string", "pattern": "^ORD-[0-9]{6}$"},
            "customer": {"$ref": "#/$defs/customer"},
            "items": {
                "type": "array", "minItems": 1, "maxItems": 100,
                "items": {"$ref": "#/$defs/line"}
            },
            "delivery": {"enum": ["shipping", "pickup"]},
            "address": {"$ref": "#/$defs/address"}
        },
        "if": {"properties": {"delivery": {"const": "shipping"}}},
        "then": {"required": ["address"]},
        "$defs": {
            "customer": {
                "type": "object", "required": ["id", "name"],
                "additionalProperties": false,
                "properties": {
                    "id": {"type": "integer", "minimum": 1},
                    "name": {"type": "string", "minLength": 1, "maxLength": 100}
                }
            },
            "line": {
                "type": "object", "required": ["sku", "quantity", "price"],
                "additionalProperties": false,
                "properties": {
                    "sku": {"type": "string", "pattern": "^SKU-[0-9]{5}$"},
                    "quantity": {"type": "integer", "minimum": 1, "maximum": 999},
                    "price": {"type": "number", "minimum": 0, "multipleOf": 0.01}
                }
            },
            "address": {
                "type": "object", "required": ["city", "postalCode"],
                "additionalProperties": false,
                "properties": {
                    "city": {"type": "string", "minLength": 1},
                    "postalCode": {"type": "string", "pattern": "^[0-9]{5}$"}
                }
            }
        }
    })json");
    testCase.corpora[1].expectedErrors = 2;
    for (std::int64_t index = 0; index < 8; ++index)
    {
        auto valid = Parse(R"json({
            "id": "ORD-123456", "customer": {"id": 1, "name": "Example Customer"},
            "items": [], "delivery": "shipping",
            "address": {"city": "Example City", "postalCode": "12345"}
        })json");
        valid.as_object().at("customer").as_object()["id"] = index + 1;
        auto& lines = valid.as_object().at("items").as_array();
        for (std::int64_t line = 0; line < 12; ++line)
        {
            lines.emplace_back(JsonObject{{"sku", "SKU-" + std::to_string(10000 + line)},
                                          {"quantity", line + 1},
                                          {"price", 19.99 + static_cast<double>(index)}});
        }
        if (index % 2 == 0)
        {
            valid.as_object()["delivery"] = "pickup";
            valid.as_object().erase("address");
        }
        auto invalid = valid;
        auto& lastLine = invalid.as_object().at("items").as_array().back().as_object();
        lastLine["quantity"] = 0;
        lastLine["price"] = -1;
        AddPair(testCase, std::move(valid), std::move(invalid));
    }
    return testCase;
}

[[nodiscard]] JsonSchemaBenchmarkCase MakeTelemetry(std::size_t count, bool unevaluated = false)
{
    auto testCase = MakeCase("telemetry_" + std::to_string(count), R"json({
        "type": "object", "required": ["device", "samples"], "additionalProperties": false,
        "properties": {
            "device": {"type": "string", "pattern": "^device-[0-9]+$"},
            "samples": {
                "type": "array", "minItems": 1, "maxItems": 4096,
                "items": {
                    "type": "object", "required": ["timestamp", "value", "quality"],
                    "additionalProperties": false,
                    "properties": {
                        "timestamp": {"type": "integer", "minimum": 0},
                        "value": {"type": "number", "minimum": -100, "maximum": 100},
                        "quality": {"enum": ["good", "estimated", "missing"]}
                    }
                }
            }
        }
    })json");
    if (unevaluated)
    {
        testCase.name += "_unevaluated";
        auto schema = testCase.schema->Root();
        schema.as_object()["unevaluatedProperties"] = true;
        schema.as_object()["unevaluatedItems"] = true;
        testCase.schema = rad::JsonDocument::FromValue(std::move(schema));
    }
    testCase.corpora[1].expectedErrors = 8;
    for (std::int64_t device = 0; device < 4; ++device)
    {
        JsonArray samples;
        samples.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            samples.emplace_back(JsonObject{{"timestamp", 1700000000 + index},
                                            {"value", static_cast<double>(index % 100) * 0.25},
                                            {"quality", index % 4 == 0 ? "estimated" : "good"}});
        }
        JsonValue valid = JsonObject{{"device", "device-" + std::to_string(device)},
                                     {"samples", std::move(samples)}};
        auto invalid = valid;
        auto& invalidSamples = invalid.as_object().at("samples").as_array();
        for (std::size_t index = count - 8; index < count; ++index)
        {
            invalidSamples[index].as_object()["value"] = 101;
        }
        AddPair(testCase, std::move(valid), std::move(invalid));
    }
    return testCase;
}

[[nodiscard]] JsonSchemaBenchmarkCase MakeEventBatch()
{
    auto testCase = MakeCase("event_batch", R"json({
        "type": "array", "minItems": 1,
        "items": {
            "oneOf": [
                {"$ref": "#/$defs/created"},
                {"$ref": "#/$defs/updated"},
                {"$ref": "#/$defs/deleted"}
            ]
        },
        "$defs": {
            "created": {
                "type": "object", "required": ["kind", "id", "name"],
                "additionalProperties": false,
                "properties": {
                    "kind": {"const": "created"}, "id": {"type": "integer", "minimum": 1},
                    "name": {"type": "string", "minLength": 1}
                }
            },
            "updated": {
                "type": "object", "required": ["kind", "id", "revision"],
                "additionalProperties": false,
                "properties": {
                    "kind": {"const": "updated"}, "id": {"type": "integer", "minimum": 1},
                    "revision": {"type": "integer", "minimum": 1}
                }
            },
            "deleted": {
                "type": "object", "required": ["kind", "id"],
                "additionalProperties": false,
                "properties": {
                    "kind": {"const": "deleted"}, "id": {"type": "integer", "minimum": 1}
                }
            }
        }
    })json");
    for (std::int64_t batch = 0; batch < 4; ++batch)
    {
        JsonArray events;
        events.reserve(64);
        for (std::int64_t index = 0; index < 64; ++index)
        {
            JsonObject event{{"id", batch * 64 + index + 1}};
            switch (index % 3)
            {
            case 0:
                event["kind"] = "created";
                event["name"] = "record-" + std::to_string(index);
                break;
            case 1:
                event["kind"] = "updated";
                event["revision"] = index + 1;
                break;
            default:
                event["kind"] = "deleted";
                break;
            }
            events.emplace_back(std::move(event));
        }
        JsonValue valid = std::move(events);
        auto invalid = valid;
        invalid.as_array().back().as_object()["kind"] = "unknown";
        AddPair(testCase, std::move(valid), std::move(invalid));
    }
    return testCase;
}

[[nodiscard]] JsonSchemaBenchmarkCase MakeMetricLabels()
{
    auto testCase = MakeCase("metric_labels_128", R"json({
        "type": "object", "minProperties": 1, "maxProperties": 256,
        "patternProperties": {
            "^metric_[a-z]+_[0-9]+$": {"type": "number", "minimum": 0}
        },
        "additionalProperties": false
    })json");
    for (std::int64_t batch = 0; batch < 4; ++batch)
    {
        JsonObject labels;
        for (std::int64_t index = 0; index < 128; ++index)
        {
            labels["metric_latency_" + std::to_string(index)] =
                static_cast<double>(batch + index) * 0.125;
        }
        JsonValue valid = std::move(labels);
        auto invalid = valid;
        invalid.as_object()["metric_latency_127"] = -1;
        AddPair(testCase, std::move(valid), std::move(invalid));
    }
    return testCase;
}

[[nodiscard]] JsonSchemaBenchmarkCase MakeTypeValidation(std::string name, JsonValue type)
{
    auto testCase = MakeCase(std::move(name), R"json({
        "type": "array", "items": {"type": "integer"}
    })json");
    auto schema = testCase.schema->Root();
    schema.as_object().at("items").as_object()["type"] = std::move(type);
    testCase.schema = rad::JsonDocument::FromValue(std::move(schema));
    constexpr std::size_t count = 1024;
    for (std::uint64_t batch = 0; batch < 4; ++batch)
    {
        JsonArray values;
        values.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            values.emplace_back(batch * count + index);
        }
        JsonValue valid = std::move(values);
        auto invalid = valid;
        invalid.as_array().back() = JsonObject{};
        AddPair(testCase, std::move(valid), std::move(invalid));
    }
    return testCase;
}

[[nodiscard]] JsonSchemaBenchmarkCase MakeMultipleOf(std::string name, JsonValue divisor,
                                                     bool nested, bool floatingInstances)
{
    constexpr std::string_view property = "monthly/accounting~balances";
    auto testCase = MakeCase(
        std::move(name),
        nested ? R"json({"properties": {"monthly/accounting~balances":
                     {"type": "array", "items": {"multipleOf": 1}}}})json"
               : R"json({"type": "array", "items": {"multipleOf": 1}})json");
    auto schema = testCase.schema->Root();
    auto& arraySchema = nested ? schema.as_object().at("properties").as_object().at(property)
                               : schema;
    arraySchema.as_object().at("items").as_object()["multipleOf"] = divisor;
    testCase.schema = rad::JsonDocument::FromValue(std::move(schema));
    testCase.corpora[1].name = "invalid_early";
    testCase.corpora.push_back({"invalid_late", {}, 1});
    constexpr std::size_t count = 1024;
    for (std::uint64_t batch = 0; batch < 4; ++batch)
    {
        JsonArray values;
        values.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            const auto value = batch * count + index;
            if (floatingInstances)
            {
                const auto step = divisor.is_double()
                                      ? divisor.as_double()
                                      : static_cast<double>(divisor.as_int64());
                values.emplace_back(
                    Parse(std::format("{:.2f}", static_cast<double>(value) * step)).as_double());
            }
            else
            {
                values.emplace_back(value * 3);
            }
        }
        JsonValue valid = std::move(values);
        auto early = valid;
        auto late = valid;
        const JsonValue invalid = divisor.is_double() ? JsonValue(0.125)
                                  : floatingInstances ? JsonValue(1.0) : JsonValue(1);
        early.as_array().front() = invalid;
        late.as_array().back() = invalid;
        if (nested)
        {
            valid = JsonObject{{property, std::move(valid)}};
            early = JsonObject{{property, std::move(early)}};
            late = JsonObject{{property, std::move(late)}};
        }
        AddPair(testCase, std::move(valid), std::move(early));
        testCase.corpora[2].instances.push_back(std::move(late));
    }
    return testCase;
}

[[nodiscard]] JsonSchemaBenchmarkCase MakeManyDivisors()
{
    auto testCase = MakeMultipleOf("multiple_many_divisors_1024", 0.25, false, true);
    auto schema = testCase.schema->Root();
    JsonArray prefix;
    for (std::size_t index = 0; index < 256; ++index)
    {
        prefix.emplace_back(JsonObject{{"multipleOf", 0.25}});
    }
    schema.as_object()["prefixItems"] = std::move(prefix);
    testCase.schema = rad::JsonDocument::FromValue(std::move(schema));
    return testCase;
}

[[nodiscard]] JsonSchemaBenchmarkCase MakeUniqueIds(std::size_t count)
{
    auto testCase = MakeCase("unique_ids_" + std::to_string(count), R"json({
        "type": "array", "minItems": 1, "maxItems": 4096, "uniqueItems": true,
        "items": {"type": "integer", "minimum": 1}
    })json");
    for (std::uint64_t batch = 0; batch < 4; ++batch)
    {
        JsonArray ids;
        ids.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            ids.emplace_back(batch * count + index + 1);
        }
        JsonValue valid = std::move(ids);
        auto invalid = valid;
        invalid.as_array().back() = invalid.as_array().front();
        AddPair(testCase, std::move(valid), std::move(invalid));
    }
    return testCase;
}

[[nodiscard]] JsonValue MakeNode(std::uint64_t& nextId, std::size_t depth)
{
    const auto id = nextId++;
    JsonArray children;
    if (depth != 0)
    {
        for (std::size_t index = 0; index < 3; ++index)
        {
            children.push_back(MakeNode(nextId, depth - 1));
        }
    }
    return JsonObject{{"id", id}, {"children", std::move(children)}};
}

[[nodiscard]] JsonSchemaBenchmarkCase MakeTree()
{
    auto testCase = MakeCase("recursive_tree_121", R"json({
        "$ref": "#/$defs/node",
        "$defs": {
            "node": {
                "type": "object", "required": ["id", "children"],
                "additionalProperties": false,
                "properties": {
                    "id": {"type": "integer", "minimum": 1},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/node"}}
                }
            }
        }
    })json");
    for (std::uint64_t batch = 0; batch < 4; ++batch)
    {
        std::uint64_t nextId = batch * 121 + 1;
        auto valid = MakeNode(nextId, 4);
        auto invalid = valid;
        JsonValue* leaf = &invalid;
        for (std::size_t depth = 0; depth < 4; ++depth)
        {
            leaf = &leaf->as_object().at("children").as_array().back();
        }
        leaf->as_object()["id"] = 0;
        AddPair(testCase, std::move(valid), std::move(invalid));
    }
    return testCase;
}

} // namespace

std::vector<JsonSchemaBenchmarkCase> MakeJsonSchemaBenchmarkCases()
{
    return {MakeServiceConfig(), MakeOrder(),                  MakeTelemetry(32),
            MakeTelemetry(1024), MakeTelemetry(1024, true),     MakeEventBatch(),
            MakeMetricLabels(),  MakeUniqueIds(32),             MakeUniqueIds(1024),
            MakeTree(),
            MakeTypeValidation("type_single_1024", "integer"),
            MakeTypeValidation("type_union_first_1024",
                               JsonArray{"integer", "null", "boolean", "array", "string"}),
            MakeTypeValidation("type_union_last_1024",
                               JsonArray{"null", "boolean", "array", "string", "integer"}),
            MakeMultipleOf("multiple_integer_1024", 3, false, false),
            MakeMultipleOf("multiple_decimal_1024", 0.25, false, true),
            MakeMultipleOf("multiple_nested_decimal_1024", 0.25, true, true),
            MakeMultipleOf("multiple_integer_instances_1024", 0.25, false, false),
            MakeMultipleOf("multiple_integer_divisor_1024", 3, false, true),
            MakeMultipleOf("multiple_cents_1024", 0.01, false, true),
            MakeManyDivisors()};
}

} // namespace benchmark
