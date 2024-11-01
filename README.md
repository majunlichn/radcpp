# radcpp
Great C++ collections.

## Build

1. Setup [microsoft/vcpkg: C++ Library Manager for Windows, Linux, and MacOS](https://github.com/microsoft/vcpkg)

   - Set environment variable "VKPKG_ROOT" as the absolute path of vcpkg repository.

2. Install the following packages:

   - boost
   - fmt
   - spdlog
   - backward-cpp
   - cpu-features
   - minizip-ng[core,zstd,zlib,wzaes,pkcrypt,lzma,bzip2]

3. Call cmake to generate project files:

   `cmake -S . -B build -DENABLE_ASAN=ON`
