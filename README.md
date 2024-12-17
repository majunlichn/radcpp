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
   - gtest
   - eigen3

3. Make sure all submodules updated:

   `git submodule update --init --recursive`

4. Call cmake to generate project files:

   `cmake -S . -B build -DENABLE_ASAN=ON`

Optionally, you can use [radsdk](https://github.com/majunlichn/radsdk) to setup dependencies:

```powershell
# Clone radsdk (requires Git LFS support):
git clone https://github.com/majunlichn/radsdk.git
$RADSDK_PATH="/path/to/radsdk"
# Execute setup.py to download and build additional libraries:
cd /path/to/radsdk
python setup.py
# Generate project files and build:
cd /path/to/radcpp
cmake -S . -B build -D VCPKG_MANIFEST_DIR=$RADSDK_PATH -D VCPKG_INSTALLED_DIR=$RADSDK_PATH/vcpkg_installed
```
