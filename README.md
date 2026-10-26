# rad

A simple, practical, and high-performance C++ foundational library.

## Source layout

- Public headers, their implementation files, and API tests live together under
  `src/rad/<Module>/`. Public includes use paths such as `<rad/IO/JsonSchema.h>`.
- Larger features keep library-only implementation headers and sources together in
  module-local directories such as [IO/Private/JsonSchema](src/rad/IO/Private/JsonSchema/).
  `Private` directories are not installed; public headers must not depend on them.
- Tests use the `.test.cpp` suffix, with shared test helpers in `.test.h` files.
  Tests of private implementation units can live beside those units. Tests that
  exercise the public API stay beside the public interface, even when split by topic.
  Test headers are not installed.

## Tests

```powershell
cmake -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```
