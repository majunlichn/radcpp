# rad

A simple, practical C++20 foundational static library, exposed as the CMake target `rad::rad`.

## Goals

- Provide reusable building blocks for application and tooling development.
- Favor clear APIs, practical utility, and efficient implementations over a large framework.
- Make resource ownership, memory control, and development diagnostics accessible.

## Features

- **Container**
  - `SmallVector` for small sequences.

- **Core**
  - Strings, Unicode, formatting, and Base64.
  - Hashing, CRCs, random values, numeric utilities, and reduced-precision types.
  - Reference counting, allocation, and memory tracking.

- **Diagnostics**
  - Exceptions with source locations and captured stack traces.

- **IO**
  - Binary and text file I/O, with configurable open modes and atomic whole-file replacement.
  - `FileCleaner` for automatic file cleanup, with thread-safe registration.
  - JSON parsing and formatting.
  - JSON Schema compilation and validation (Draft 7, 2019-09, and 2020-12 subsets).
  - Image loading, encoding, conversion, and resizing.
  - Logging.

- **Math**
  - Vectors, matrices, quaternions, and geometry helpers.
  - 3D transforms and projections.

- **System** (platform-specific Windows/POSIX behavior)
  - Application initialization, filesystem, and OS utilities.
  - Process output capture and thread utilities.
  - Time, stopwatches, deadlines, and CPU feature detection.

## Build and test

- Requires CMake 3.25+, a C++20 compiler, and the dependencies in [vcpkg.json](vcpkg.json).
- Set `VCPKG_ROOT` to use its toolchain automatically; the first configure installs dependencies.
- Tests are enabled by default and download the official JSON Schema test suite during
  configuration, requiring Git and network access.

**Single-config** (Ninja, Unix Makefiles), from the repository root using Bash:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

**Multi-config** (Visual Studio, Ninja Multi-Config): omit
`-DCMAKE_BUILD_TYPE=Debug`; add `--config Debug` when building and `-C Debug` when running CTest.
Use the same configuration throughout.

**Runtime outputs** are in `build/bin/` for single-config builds or `build/bin/<Config>/` for
multi-config builds. When running the test executable directly, use that directory as the
working directory to keep logs, temporary files, and other test outputs out of the source tree.

## Benchmarks

- [JsonSchemaBenchmark](benchmarks/): measures JSON Schema compilation and validation performance.
  Use `--filter child_path` for nested objects with long property names, with and without JSON
  Pointer escapes.
  Use `--filter metric_label_batch` to compare `patternProperties` with and without
  `additionalProperties` or `unevaluatedProperties`.
