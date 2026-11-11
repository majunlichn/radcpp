# Agent guide

`rad` is a C++20 foundational static library, exposed as `rad::rad`.
See [README.md](README.md) for project goals and features.

## Source layout

- `src/rad/<Module>/`: library code, grouped by module.
  - Keep public `Foo.h`, optional `Foo.cpp`, and API tests (`Foo*.test.cpp`) together.
  - Group feature internals and their tests under `<Module>/Private/<Feature>/`, following
    [JsonSchema](src/rad/IO/Private/JsonSchema/).
  - Public headers must not depend on private headers. `Private/` and shared test helpers
    (`*.test.h`) are not installed.
  - The public include root is `src`, e.g. `<rad/IO/JsonSchema.h>`.
- `cmake/`: build support and generation templates.
- `thirdparty/`: vendored code and data. Declare other dependencies in [vcpkg.json](vcpkg.json).
- Register files explicitly in `RAD_SOURCES` or `RAD_TEST_SOURCES` in
  [CMakeLists.txt](CMakeLists.txt). Do not use source globbing; unlisted files are not built.

## C++ style

- Follow [.clang-format](.clang-format): Allman braces, four-space indentation, 100-column limit.
  Preserve include order; do not sort includes.
- Use PascalCase types/functions, `m_` members, `#pragma once`, `[[nodiscard]]` on non-void
  returners, and closing namespace/class comments.
- Keep compiler flags and platform defines centralized in [CMakeLists.txt](CMakeLists.txt) and
  [radCompileOptions.cmake](cmake/radCompileOptions.cmake); do not add per-file workarounds.

## Build and test

- Follow the [README build and test instructions](README.md#build-and-test).
- Update adjacent tests for behavior changes and run affected suites.
- **Direct GTest runs**: use the **runtime output directory** as the working directory, not the
  repository root, to keep logs, temporary files, and other test outputs out of the source tree.
  - **Multi-config**: **`build/bin/<Config>/`**, matching the built configuration.
  - **Single-config**: `build/bin/`.

## Test conventions

- Use `TEST(Suite, Name)`: `Suite` names the tested type/unit (usually file stem, not module);
  `Name` describes behavior without repeating the suite, e.g. `Base64.Encode`. Use PascalCase
  without underscores or redundant `Test`/`Tests` suffixes.
- Keep `Suite.Name` unique across the executable; use distinct suites for independent types
  and for private implementation tests.
- Reuse [TestMain.cpp](src/rad/TestMain.cpp) for `main` and `rad::Application` initialization;
  do not add another entry point.
