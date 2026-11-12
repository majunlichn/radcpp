#include "JsonSchemaBenchmarkCases.h"

#include <rad/System/Application.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

namespace
{

using Clock = std::chrono::steady_clock;
using benchmark::JsonSchemaBenchmarkCase;
using benchmark::JsonSchemaBenchmarkCorpus;

struct Options
{
    std::string output = "JsonSchemaBenchmark.csv";
    std::string filter;
    std::uint32_t samples = 5;
    std::uint32_t minTimeMs = 100;
    bool help = false;
    bool list = false;
}; // struct Options

struct Measurement
{
    std::chrono::nanoseconds elapsed;
    std::uint64_t operations;
}; // struct Measurement

[[nodiscard]] std::uint32_t ParsePositive(std::string_view text, std::string_view option)
{
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0)
    {
        throw std::runtime_error(std::string(option) + " requires a positive 32-bit integer");
    }
    return value;
}

[[nodiscard]] Options ParseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string_view option = argv[index];
        if (option == "--help")
        {
            options.help = true;
            continue;
        }
        if (option == "--list")
        {
            options.list = true;
            continue;
        }
        if (option != "--output" && option != "--filter" && option != "--samples" &&
            option != "--min-time-ms")
        {
            throw std::runtime_error("Unknown option: " + std::string(option));
        }
        if (++index == argc)
        {
            throw std::runtime_error("Missing value for " + std::string(option));
        }
        const std::string_view value = argv[index];
        if (option == "--output")
        {
            options.output = value;
        }
        else if (option == "--filter")
        {
            options.filter = value;
        }
        else if (option == "--samples")
        {
            options.samples = ParsePositive(value, option);
        }
        else
        {
            options.minTimeMs = ParsePositive(value, option);
        }
    }
    return options;
}

void PrintHelp()
{
    std::cout << "JsonSchemaBenchmark\n"
                 "  --output PATH      CSV output, overwritten (default: JsonSchemaBenchmark.csv)\n"
                 "  --filter TEXT      Run cases whose names contain TEXT\n"
                 "  --samples N        Repetitions averaged per result (default: 5)\n"
                 "  --min-time-ms N    Calibration target per sample in ms (default: 100)\n"
                 "  --list             List matching cases without measuring or writing CSV\n"
                 "  --help             Show this help\n"
                 "Timing: us_per_op is mean microseconds per compilation or validated instance.\n"
                 "Results are printed to the console and saved to CSV.\n";
}

[[nodiscard]] rad::JsonSchema Compile(const JsonSchemaBenchmarkCase& testCase)
{
    auto result = rad::JsonSchema::Compile(testCase.schema);
    if (!result)
    {
        const auto& error = result.error();
        throw std::runtime_error(testCase.name + ": compilation failed at " + error.schemaPath +
                                 ": " + error.message);
    }
    return std::move(result).value();
}

void CheckValidation(const JsonSchemaBenchmarkCase& testCase,
                     const JsonSchemaBenchmarkCorpus& corpus,
                     const rad::JsonSchemaValidationResult& result, std::size_t maxErrors)
{
    const auto expectedStatus = corpus.expectedErrors >= maxErrors
                                    ? rad::JsonSchemaValidationStatus::MaxErrorsReached
                                    : rad::JsonSchemaValidationStatus::Completed;
    if (result.status != expectedStatus ||
        result.errors.size() != std::min(corpus.expectedErrors, maxErrors) ||
        static_cast<bool>(result) != (corpus.expectedErrors == 0))
    {
        std::string message = testCase.name + "/" + corpus.name +
                              ": unexpected validation result (status=" +
                              std::to_string(static_cast<int>(result.status)) +
                              ", errors=" + std::to_string(result.errors.size()) + ")";
        for (const auto& error : result.errors)
        {
            message +=
                "\n  " + error.instancePath + " at " + error.schemaPath + ": " + error.message;
        }
        throw std::runtime_error(message);
    }
}

template <typename Batch>
[[nodiscard]] Measurement Measure(Batch& batch, std::uint64_t iterations)
{
    std::uint64_t operations = 0;
    const auto start = Clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index)
    {
        const auto current = batch();
        if (current > std::numeric_limits<std::uint64_t>::max() - operations)
        {
            throw std::runtime_error("Benchmark operation count overflowed");
        }
        operations += current;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);
    return {elapsed, operations};
}

void WriteCsvText(std::ostream& output, std::string_view text)
{
    output << '"';
    for (const char character : text)
    {
        if (character == '"')
        {
            output << '"';
        }
        output << character;
    }
    output << '"';
}

template <typename Batch>
void Run(std::ostream& csv, const Options& options, const JsonSchemaBenchmarkCase& testCase,
         std::string_view phase, std::string_view corpus, std::size_t maxErrors, Batch batch)
{
    std::uint64_t iterations = 1;
    const auto target = std::chrono::milliseconds(options.minTimeMs);
    while (Measure(batch, iterations).elapsed < target)
    {
        if (iterations > std::numeric_limits<std::uint64_t>::max() / 2)
        {
            throw std::runtime_error("Calibration iteration count overflowed");
        }
        iterations *= 2;
    }

    double totalNsPerOp = 0;
    for (std::uint32_t sample = 0; sample < options.samples; ++sample)
    {
        const auto measurement = Measure(batch, iterations);
        const auto elapsedNs = measurement.elapsed.count();
        if (elapsedNs <= 0 || measurement.operations == 0)
        {
            throw std::runtime_error("Benchmark produced an empty measurement");
        }
        const auto operations = static_cast<double>(measurement.operations);
        totalNsPerOp += static_cast<double>(elapsedNs) / operations;
    }
    const auto meanUsPerOp = totalNsPerOp / options.samples / 1000.0;
    WriteCsvText(csv, testCase.name);
    csv << ',';
    WriteCsvText(csv, phase);
    csv << ',';
    WriteCsvText(csv, corpus);
    csv << ',' << maxErrors << ',' << meanUsPerOp << '\n';

    std::cout << "  " << phase;
    if (!corpus.empty())
    {
        std::cout << " (" << corpus << ", max_errors=" << maxErrors << ')';
    }
    std::cout << std::format(": {:.2f} us/op ({} iterations/sample)\n", meanUsPerOp, iterations);
}

int RunBenchmark(const Options& options)
{
    auto cases = benchmark::MakeJsonSchemaBenchmarkCases();
    std::erase_if(cases, [&options](const auto& testCase)
                  { return testCase.name.find(options.filter) == std::string::npos; });
    if (cases.empty())
    {
        throw std::runtime_error("No benchmark cases match filter: " + options.filter);
    }
    if (options.list)
    {
        for (const auto& testCase : cases)
        {
            std::cout << testCase.name << '\n';
        }
        return 0;
    }
    std::cout << "JsonSchemaBenchmark\n"
              << "  Build: " << RAD_BENCHMARK_BUILD_CONFIG << '\n'
              << "  Output: " << options.output << " (overwrite)\n"
              << "  Cases: " << cases.size() << '\n'
              << "  Samples per result: " << options.samples << '\n'
              << "  Calibration target: " << options.minTimeMs << " ms/sample\n";
    if (std::string_view(RAD_BENCHMARK_BUILD_CONFIG) != "Release")
    {
        std::cerr << "Warning: build configuration is " << RAD_BENCHMARK_BUILD_CONFIG
                  << "; use Release for representative performance results.\n";
    }

    // Check every selected corpus before opening the result file or starting the timers.
    std::cout << "Checking fixture validity...\n";
    for (const auto& testCase : cases)
    {
        const auto schema = Compile(testCase);
        for (const auto& corpus : testCase.corpora)
        {
            if (corpus.instances.empty())
            {
                throw std::runtime_error(testCase.name + "/" + corpus.name + ": empty corpus");
            }
            for (const std::size_t maxErrors : {std::size_t{1}, std::size_t{64}})
            {
                rad::JsonSchemaValidationOptions validation;
                validation.maxErrors = maxErrors;
                for (const auto& instance : corpus.instances)
                {
                    CheckValidation(testCase, corpus, schema.Validate(instance, validation),
                                    maxErrors);
                }
            }
        }
    }

    std::cout << "All selected fixtures passed.\n";

    std::ofstream csv;
    csv.exceptions(std::ios::failbit | std::ios::badbit);
    csv.imbue(std::locale::classic());
    csv.open(options.output, std::ios::out | std::ios::trunc);
    csv << std::setprecision(12);
    csv << "case,phase,corpus,max_errors,us_per_op\n";

    std::size_t caseIndex = 0;
    std::size_t resultCount = 0;
    for (const auto& testCase : cases)
    {
        std::cout << '[' << ++caseIndex << '/' << cases.size() << "] " << testCase.name << '\n';
        Run(csv, options, testCase, "compile", "", 0,
            [&testCase]()
            {
                const auto compiled = Compile(testCase);
                return std::uint64_t{1};
            });
        ++resultCount;

        const auto schema = Compile(testCase);
        for (const auto& corpus : testCase.corpora)
        {
            for (const std::size_t maxErrors : {std::size_t{1}, std::size_t{64}})
            {
                rad::JsonSchemaValidationOptions validation;
                validation.maxErrors = maxErrors;
                Run(csv, options, testCase, "validate", corpus.name, maxErrors,
                    [&testCase, &corpus, &schema, &validation]()
                    {
                        for (const auto& instance : corpus.instances)
                        {
                            const auto result = schema.Validate(instance, validation);
                            CheckValidation(testCase, corpus, result, validation.maxErrors);
                        }
                        return static_cast<std::uint64_t>(corpus.instances.size());
                    });
                ++resultCount;
            }
        }
        csv.flush();
    }
    csv.close();
    std::cout << "Wrote " << resultCount << " results to " << options.output << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        auto options = ParseOptions(argc, argv);
        if (options.help)
        {
            PrintHelp();
            return 0;
        }
        rad::Application::Instance().Init(argc, argv);
        if (!options.list)
        {
            options.output = std::filesystem::absolute(options.output).string();
        }
        try
        {
            return RunBenchmark(options);
        }
        catch (const std::ios_base::failure& error)
        {
            throw std::runtime_error(
                "CSV output failed for '" + options.output + "': " + error.what() +
                "\nCheck that the parent directory exists and is writable, and close the CSV "
                "in Excel or other programs that may lock it. Alternatively, use --output "
                "with a different file name. A failed write may leave a partial CSV.");
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "JsonSchemaBenchmark failed: " << error.what() << '\n';
        return 1;
    }
}
