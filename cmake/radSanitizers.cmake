include_guard(GLOBAL)

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_MSAN "Enable MemorySanitizer" OFF)
option(ENABLE_LSAN "Enable LeakSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)

if(ENABLE_TSAN AND (ENABLE_ASAN OR ENABLE_LSAN))
    message("ThreadSanitizer cannot be used with AddressSanitizer or LeakSanitizer!")
    set(ENABLE_ASAN OFF)
    set(ENABLE_LSAN OFF)
endif()

# AddressSanitizer
if(ENABLE_ASAN)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/fsanitize=address)
        add_link_options(/INCREMENTAL:NO)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-ipa-icf)
        add_link_options(-fsanitize=address)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # https://clang.llvm.org/docs/AddressSanitizer.html
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls)
        add_link_options(-fsanitize=address)
    else()
        message("AddressSanitizer is not supported by current compiler!")
    endif()
endif()

# MemorySanitizer
if(ENABLE_MSAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # https://clang.llvm.org/docs/MemorySanitizer.html
        add_compile_options(-fsanitize=memory -fno-omit-frame-pointer -fno-optimize-sibling-calls)
        add_link_options(-fsanitize=memory)
    else()
        message("MemorySanitizer is not supported by current compiler!")
    endif()
endif()

# LeakSanitizer
if(ENABLE_LSAN)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
        add_link_options(-fsanitize=leak)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # https://clang.llvm.org/docs/LeakSanitizer.html
        add_link_options(-fsanitize=leak)
    else()
        message("LeakSanitizer is not supported by current compiler!")
    endif()
endif()

# UndefinedBehaviorSanitizer
if(ENABLE_UBSAN)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
        add_compile_options(-fsanitize=undefined)
        add_link_options(-fsanitize=undefined)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
        add_compile_options(-fsanitize=undefined)
        add_link_options(-fsanitize=undefined)
    else()
        message("UndefinedBehaviorSanitizer is not supported by current compiler!")
    endif()
endif()

# ThreadSanitizer: https://github.com/google/sanitizers/wiki/ThreadSanitizerFlags
if(ENABLE_TSAN)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
        add_compile_options(-fsanitize=thread)
        add_link_options(-fsanitize=thread)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # https://clang.llvm.org/docs/ThreadSanitizer.html
        add_compile_options(-fsanitize=thread)
        add_link_options(-fsanitize=thread)
    else()
        message("ThreadSanitizer is not supported by current compiler!")
    endif()
endif()
