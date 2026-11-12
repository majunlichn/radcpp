#pragma once

#include <rad/IO/JsonSchema.h>

#include <cstddef>
#include <string>
#include <vector>

namespace benchmark
{

struct JsonSchemaBenchmarkCorpus
{
    std::string name;
    std::vector<rad::JsonValue> instances;
    // Exact number of violations per instance when validation completes without a limit.
    std::size_t expectedErrors = 0;
}; // struct JsonSchemaBenchmarkCorpus

struct JsonSchemaBenchmarkCase
{
    std::string name;
    rad::Ref<const rad::JsonDocument> schema;
    std::vector<JsonSchemaBenchmarkCorpus> corpora;
}; // struct JsonSchemaBenchmarkCase

[[nodiscard]] std::vector<JsonSchemaBenchmarkCase> MakeJsonSchemaBenchmarkCases();

} // namespace benchmark
