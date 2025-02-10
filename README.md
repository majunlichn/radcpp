# radcpp

Great C++ collections.

## Build

1. Make sure all submodules are updated:

    ```powershell
    git submodule update --init --recursive
    ```

2. Setup [microsoft/vcpkg](https://github.com/microsoft/vcpkg):

    ```powershell
    # Clone vcpkg into a folder you like, which can also be shared by other projects:
    git clone https://github.com/microsoft/vcpkg.git
    # Run the bootstrap script:
    cd vcpkg
    .\bootstrap-vcpkg.bat # Linux: ./bootstrap-vcpkg.sh
    # Configure the VCPKG_ROOT environment variable for convenience (use "/" as the directory separator):
    $env:VCPKG_ROOT="C:/path/to/vcpkg" # Linux: export VCPKG_ROOT="/path/to/vcpkg"
    ```

3. Install the following vcpkg packages:

    - backward-cpp
    - boost
    - cpu-features
    - eigen3
    - fmt
    - glm
    - gtest
    - minizip-ng
    - spdlog

    For example, to install boost on Windows (classic mode):

    ```powershell
    # At the root of vcpkg repository:
    .\vcpkg.exe install boost:x64-windows # Linux: ./vcpkg install boost
    ```

    You can also add the packages to your `vcpkg.json` (manifest mode), please refer to https://learn.microsoft.com/en-us/vcpkg/get_started/get-started.

4. Call CMake to generate project files and build:

    ```powershell
    # To enable address sanitizer: -D ENABLE_ASAN=ON
    cmake -S . -B build
    # Note that CMake will add the vcpkg toolchain file automatically if you set environment variable VCPKG_ROOT. If you want to set it explicitly:
    cmake -S . -B build -D CMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
    # To build with radsdk (https://github.com/majunlichn/radsdk):
    cmake -S . -B build -D VCPKG_MANIFEST_DIR="$env:RADSDK_ROOT" -D VCPKG_INSTALLED_DIR="$env:RADSDK_ROOT/vcpkg_installed"
    ```
