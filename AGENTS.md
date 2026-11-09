# AGENTS.md

`rad` is a C++20 foundational static library (`rad::rad`). Single package; no CI.

## Layout

- `src/rad/<Module>/` — modules: `Container`, `Core`, `Diagnostics`, `IO`, `Math`, `System`. Public include root is `src/`, so code includes `<rad/Core/String.h>`.
- Each unit is `Foo.h` (+ `Foo.cpp` when not header-only) with tests in `Foo.test.cpp` beside it.
- `thirdparty/pcg-cpp` is vendored; all other deps come from vcpkg.

## Build & test (Windows, MSVC, multi-config generator)

- vcpkg manifest mode (`vcpkg.json`). Configure auto-uses `%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake`; first configure installs all deps (slow).
- Configure: `cmake -B build` — build: `cmake --build build --config Debug`.
- Visual Studio (MSVC) multi-config builds put binaries in `build\bin\<Config>\`, such as `build\bin\Debug\` or `build\bin\Release\`. Use the directory matching the configuration that was built.
- Direct GTest runs must use the executable's output directory as the working directory, **not the repository root**, so logs and other relative-path outputs do not pollute the source tree. Specifying an executable's path does not change its working directory.
  From the repository root: `Push-Location .\build\bin\Debug; try { .\rad_tests.exe --gtest_filter=Core.EncodeBase64 } finally { Pop-Location }`. Substitute `Release` when testing a Release build; omit the filter for the full suite.
- Alternatively: `ctest --test-dir build -C Debug --output-on-failure`. `-C` is required with the Visual Studio generator and must match the built configuration (`Debug` or `Release`).
- CMake options: `RAD_BUILD_TESTS` (ON), `RAD_ENABLE_INSTALL` (ON), `ENABLE_ASAN` (OFF), `RAD_ENABLE_MEMORY_TRACKING` (defaults to 1 in non-NDEBUG builds).

## Conventions

- **No source globbing**: add new files manually to `RAD_SOURCES` / `RAD_TEST_SOURCES` in `CMakeLists.txt` or they are silently not built.
- Tests use `TEST(Suite, Name)` where Suite is the module (`Core`, `IO`, ...). `src/rad/TestMain.cpp` provides `main` (initializes `rad::Application`) — never add another.
- `.clang-format`: LLVM base, **Allman braces**, 4-space indent, 100-col limit, and `SortIncludes: false` — do NOT sort/reorder includes; match the surrounding order.
- Style: PascalCase types/functions, `m_` member prefix, `#pragma once`, `[[nodiscard]]` on non-void returners, closing-namespace/class comments (`} // namespace rad`).
- MSVC flags `/utf-8`, `/Zc:preprocessor`, `/permissive-` and `NOMINMAX` / `_WIN32_WINNT=0x0A00` are already set by the build; don't add per-file workarounds.
