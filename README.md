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
    # Configure the VCPKG_ROOT environment variable for convenience:
    $env:VCPKG_ROOT="C:\path\to\vcpkg" # Linux: export VCPKG_ROOT="/path/to/vcpkg"
    ```

3. Install the following vcpkg packages:

    - boost
    - fmt
    - spdlog
    - backward-cpp
    - cpu-features
    - minizip-ng[core,zstd,zlib,wzaes,pkcrypt,lzma,bzip2]
    - gtest
    - eigen3
    
    For example, in vcpkg folder call `.\vcpkg.exe install boost:x64-windows` to install boost on Windows (classic mode).
    You can also add the dependencies to your `vcpkg.json` (manifest mode, please refer to: https://learn.microsoft.com/en-us/vcpkg/consume/manifest-mode).

4. Call CMake to generate project files and build:

    ```powershell
    cmake -S . -B build -D ENABLE_ASAN=ON
    ```
