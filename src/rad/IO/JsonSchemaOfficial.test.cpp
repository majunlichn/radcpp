#include <rad/IO/File.h>
#include <rad/IO/JsonSchema.h>
#include <rad/System/OS.h>

#include "JsonSchema.test.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <vector>
#include <iostream>
#include <cstddef>

using rad::test::FormatValidationErrors;

namespace
{

[[nodiscard]] bool IsKnownInvalidOfficialSchema(const rad::JsonValue& schema)
{
    if (!schema.is_object())
    {
        return false;
    }
    const auto* enumValue = schema.as_object().if_contains("enum");
    return enumValue != nullptr && enumValue->is_array() && enumValue->as_array().empty();
}

} // namespace

TEST(IO, JsonSchemaOfficialTestSuite)
{
    const auto suiteRoot = rad::os::executable_directory() / "JSON-Schema-Test-Suite";
    ASSERT_TRUE(std::filesystem::is_directory(suiteRoot))
        << "JSON Schema test suite not found: " << suiteRoot.string()
        << ". Rebuild rad_tests to copy the test data.";

    rad::JsonSchemaCompileOptions compileOptions;
    const auto remotes = suiteRoot / "remotes";
    ASSERT_TRUE(std::filesystem::is_directory(remotes)) << remotes.string();
    for (const auto& entry : std::filesystem::recursive_directory_iterator(remotes))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".json")
        {
            continue;
        }
        const auto text = rad::File::ReadAllText(entry.path());
        ASSERT_TRUE(text) << entry.path().string();
        const auto schema = rad::ParseJson(*text);
        ASSERT_TRUE(schema) << entry.path().string();
        compileOptions.documents.push_back({
            "http://localhost:1234/" +
                std::filesystem::relative(entry.path(), remotes).generic_string(),
            schema.value(),
        });
    }

    struct Suite
    {
        rad::JsonSchemaDialect dialect;
        std::string_view directory;
    };
    constexpr Suite suites[] = {
        {rad::JsonSchemaDialect::Draft7, "draft7"},
        {rad::JsonSchemaDialect::Draft2019_09, "draft2019-09"},
        {rad::JsonSchemaDialect::Draft2020_12, "draft2020-12"},
    };

    std::size_t availableSuites = 0;
    std::size_t executedCases = 0;
    for (const auto& suite : suites)
    {
        compileOptions.dialect = suite.dialect;
        const auto testsDirectory = suiteRoot / "tests" / suite.directory;
        if (!std::filesystem::is_directory(testsDirectory))
        {
            GTEST_LOG_(WARNING) << "Schema tests not found under "
                                << testsDirectory.string();
            continue;
        }
        ++availableSuites;
        std::size_t suiteExecutedCases = 0;
        std::size_t suiteReferenceCases = 0;
        std::size_t suiteSchemaRejectionChecks = 0;
        std::size_t failedCompileGroups = 0;

        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(testsDirectory))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".json")
            {
                files.push_back(entry.path());
            }
        }
        for (const auto* name : {"ecmascript-regex.json", "non-bmp-regex.json",
                                 "float-overflow.json"})
        {
            const auto file = testsDirectory / "optional" / name;
            if (std::filesystem::is_regular_file(file))
            {
                files.push_back(file);
            }
        }
        std::sort(files.begin(), files.end());

        for (const auto& file : files)
        {
            SCOPED_TRACE(file.string());
            const bool referenceFile = file.filename() == "ref.json";
            const auto text = rad::File::ReadAllText(file);
            if (!text)
            {
                ADD_FAILURE() << "Unable to read test file";
                continue;
            }
            const auto document = rad::ParseJson(*text);
            if (!document || !document.value().is_array())
            {
                ADD_FAILURE() << "Unable to parse test file";
                continue;
            }

            for (const auto& groupValue : document.value().as_array())
            {
                ASSERT_TRUE(groupValue.is_object());
                const auto& group = groupValue.as_object();
                const auto& description = group.at("description").as_string();
                SCOPED_TRACE(std::string(description.data(), description.size()));

                const auto compiled = rad::JsonSchema::Compile(group.at("schema"), compileOptions);
                if (IsKnownInvalidOfficialSchema(group.at("schema")))
                {
                    ++suiteSchemaRejectionChecks;
                    if (compiled)
                    {
                        ADD_FAILURE() << "empty enum schema must be rejected";
                    }
                    else
                    {
                        EXPECT_EQ(compiled.error().code,
                                  rad::JsonSchemaCompileErrorCode::InvalidSchema);
                        EXPECT_EQ(compiled.error().schemaPath, "/enum");
                    }
                    continue;
                }
                if (!compiled)
                {
                    ++failedCompileGroups;
                    ADD_FAILURE() << compiled.error().schemaUri
                                  << compiled.error().schemaPath << ": "
                                  << compiled.error().message;
                    continue;
                }

                for (const auto& testValue : group.at("tests").as_array())
                {
                    const auto& test = testValue.as_object();
                    const auto& testDescription = test.at("description").as_string();
                    SCOPED_TRACE(
                        std::string(testDescription.data(), testDescription.size()));

                    const bool expected = test.at("valid").as_bool();
                    const auto result = compiled.value().Validate(test.at("data"));
                    ++suiteExecutedCases;
                    ++executedCases;
                    if (referenceFile)
                    {
                        ++suiteReferenceCases;
                    }
                    EXPECT_EQ(static_cast<bool>(result), expected)
                        << std::format("data: {}\n{}",
                                       rad::PrettyJson(test.at("data")),
                                       FormatValidationErrors(result));
                    rad::JsonSchemaValidationOptions limitedOptions;
                    limitedOptions.maxErrors = 1;
                    const auto limited = compiled.value().Validate(test.at("data"), limitedOptions);
                    EXPECT_EQ(static_cast<bool>(limited), expected)
                        << FormatValidationErrors(limited);
                    EXPECT_LE(limited.errors.size(), 1);
                }
            }
        }
        EXPECT_GT(suiteExecutedCases, 0) << testsDirectory.string();
        EXPECT_GT(suiteReferenceCases, 0) << testsDirectory.string() << "\\ref.json";
        std::cout << std::format(
            "[ SCHEMA   ] {:<12} {:>4} cases, {:>2} reference, {:>2} rejection checks, "
            "{} unexpected compile failures\n",
            suite.directory, suiteExecutedCases, suiteReferenceCases,
            suiteSchemaRejectionChecks, failedCompileGroups);
    }

    if (availableSuites == 0)
    {
        GTEST_SKIP() << "No supported schema test directories were found";
    }
    EXPECT_GT(executedCases, 0);
}
