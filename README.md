# rad

A simple, practical C++20 foundational static library, exposed as the CMake target `rad::rad`.

## Goals

- Provide reusable building blocks for application and tooling development.
- Favor clear APIs, practical utility, and efficient implementations over a large framework.
- Make resource ownership, memory control, and development diagnostics accessible.

## Style

- C++20 APIs in the `rad` namespace, with PascalCase naming and standard-library integration.
- RAII and move semantics for owned resources; explicit allocation and alignment where needed.
- Build on established libraries such as Boost, Eigen, Imath, and spdlog rather than
  reimplementing every primitive.
- Use results, optional values, exceptions, or structured diagnostics as appropriate to each API.

## Features

- **Container**
  - `SmallVector`, backed by Boost.Container.

- **Core**
  - Strings, Unicode, formatting, and Base64.
  - Hashing, CRCs, random values, numeric utilities, and reduced-precision types.
  - Reference counting, allocation, and memory tracking.

- **Diagnostics**
  - Exceptions with source locations and captured stack traces.

- **IO**
  - Binary/text file I/O, JSON parsing and formatting.
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
