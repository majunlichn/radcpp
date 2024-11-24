include_guard(GLOBAL)

if(WIN32 AND MSVC)
    # If use static crt, remember to install vcpkg packages with triplet that sets VCPKG_CRT_LINKAGE=static, or there will be link errors.
    option(USE_STATIC_CRT "Link against the static runtime libraries." OFF)
    if (${USE_STATIC_CRT})
        add_compile_options(
            $<$<CONFIG:>:/MT>
            $<$<CONFIG:Debug>:/MTd>
            $<$<CONFIG:Release>:/MT>
        )
    else()
        add_compile_options(
            $<$<CONFIG:>:/MD>
            $<$<CONFIG:Debug>:/MDd>
            $<$<CONFIG:Release>:/MD>
        )
    endif()

    # Build with multiple processes: https://learn.microsoft.com/en-us/cpp/build/reference/mp-build-with-multiple-processes
    add_compile_options(/MP)
endif()
