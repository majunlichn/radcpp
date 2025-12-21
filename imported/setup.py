import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from zipfile import ZipFile

script_root = os.path.dirname(os.path.realpath(__file__))

def chdir(path: str):
    os.chdir(path)
    print("Working dir:", os.getcwd())

def run_shell(command : str, env = os.environ):
    print("Execute:", command)
    subprocess.run(command, shell=True, env=env)

def remove_dir(dir : str):
    if os.path.isdir(dir):
        shutil.rmtree(dir)

def download_file(url, filename):
    print(f"Downloading '{filename}' from '{url}'")
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

def extract_zip(filename, path="."):
    if not os.path.exists(path):
        os.makedirs(path)
    with ZipFile(filename, "r") as zip:
        print(f"Extracting '{filename}' to '{os.path.realpath(path)}'")
        zip.extractall(path)

def download_and_extract_zip(url, filename, extract_path="."):
    download_file(url, filename)
    extract_zip(filename, extract_path)

def build_SDL():
    if not os.path.exists("SDL"):
        run_shell("git clone https://github.com/libsdl-org/SDL.git")
    chdir("SDL")
    run_shell("git clean -xdf")
    run_shell("git fetch --all")
    run_shell("git checkout a864dcac25f8d6aa1991a24642ca04d9a90c5fc6")
    run_shell("git submodule update --init")
    run_shell(f"cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=build/installed")
    run_shell(f"cmake --build build --target install --config Release")

def build_SDL_mixer():
    if not os.path.exists("SDL_mixer"):
        run_shell("git clone https://github.com/libsdl-org/SDL_mixer.git")
    chdir("SDL_mixer")
    run_shell("git clean -xdf")
    run_shell("git fetch --all")
    run_shell("git checkout 5cdf029bae982df1d6c210f915fc151a616d982f")
    run_shell("git submodule update --init")
    sdl3_dir = script_root + f"/SDL/build/installed"
    run_shell(f"cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=build/installed",
                   env=dict(os.environ, SDL3_DIR=sdl3_dir))
    run_shell(f"cmake --build build --target install --config Release")

# https://github.com/KhronosGroup/KTX-Software/blob/main/BUILDING.md
def build_libktx():
    if not os.path.exists("KTX-Software"):
        run_shell("git clone https://github.com/KhronosGroup/KTX-Software.git")
    chdir("KTX-Software")
    run_shell("git clean -xdf")
    run_shell("git fetch --all")
    run_shell("git checkout 5a07bc6f8eb95b6ea5b636903b335947d4684cef")
    chdir("lib")
    run_shell(f"cmake -S . -B build -D CMAKE_INSTALL_PREFIX=build/installed")
    run_shell(f"cmake --build build --target install --config Release")

def setup_windows(tasks):
    if "mysql" in tasks:
        download_and_extract_zip(url="https://dev.mysql.com/get/Downloads/Connector-C++/mysql-connector-c++-9.1.0-winx64.zip",
                                 filename="mysql-connector-c++-9.1.0-winx64.zip")
    if "slang" in tasks:
        download_and_extract_zip(url="https://github.com/shader-slang/slang/releases/download/v2025.10/slang-2025.10-windows-x86_64.zip",
                                 filename="slang.zip",
                                 extract_path="slang")

def setup_linux(tasks):
    if "slang" in tasks:
        download_and_extract_zip(url="https://github.com/shader-slang/slang/releases/download/v2025.24.2/slang-2025.24.2-linux-x86_64-glibc-2.27.zip",
                                 filename="slang.zip",
                                 extract_path="slang")

def main() -> int:
    tasks = sys.argv[1:]
    print(f"Tasks: {tasks}")
    err = 0
    try:
        original_working_dir = os.getcwd()
        if "SDL" in tasks:
            chdir(script_root)
            build_SDL()
        if "SDL_mixer" in tasks:
            chdir(script_root)
            build_SDL_mixer()
        chdir(script_root)
        if "libktx" in tasks:
            chdir(script_root)
            build_libktx()
        chdir(script_root)
        if platform.system() == "Windows":
            chdir(script_root)
            setup_windows(tasks)
        if platform.system() == "Linux":
            chdir(script_root)
            setup_linux(tasks)
        chdir(script_root)

    except Exception as e:
        print(e)
        err = -1
    finally:
        chdir(original_working_dir)
    return err

if __name__ == '__main__':
    sys.exit(main())
