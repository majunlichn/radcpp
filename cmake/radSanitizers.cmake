include_guard()

option(ENABLE_ASAN "Enable AddressSanitizer (ASan)" OFF)

if(ENABLE_ASAN)
    if(MSVC)
        # https://learn.microsoft.com/en-us/cpp/sanitizers/asan
        add_compile_options(/fsanitize=address /Zi)
        add_link_options(/INCREMENTAL:NO)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # https://clang.llvm.org/docs/AddressSanitizer.html
        # https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html#index-fsanitize_003daddress
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
        add_link_options(-fsanitize=address)
    else()
        message(WARNING "ENABLE_ASAN is not supported for ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()
