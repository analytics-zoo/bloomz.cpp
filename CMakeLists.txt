cmake_minimum_required(VERSION 3.12...3.22)

project("bloom.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()


#
# Option list
#

# instruction set specific
option(BLOOM_AVX                    "BLOOM: enable AVX"                                     ON)
option(BLOOM_AVX2                   "BLOOM: enable AVX2"                                    ON)
option(BLOOM_AVX512                 "BLOOM: enable AVX512"                                  OFF)
option(BLOOM_AVX512_VBMI            "BLOOM: enable AVX512-VBMI"                             OFF)
option(BLOOM_AVX512_VNNI            "BLOOM: enable AVX512-VNNI"                             OFF)
option(BLOOM_FMA                    "BLOOM: enable FMA"                                     ON)

#
# Build info header
#

# Generate initial build-info.h

#
# Compile flags
#

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            # Apple M1, M2, etc.
            # Raspberry Pi 3, 4, Zero 2 (64-bit)
            add_compile_options(-mcpu=native)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
            # Raspberry Pi 1, Zero
            add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
            # Raspberry Pi 2
            add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
            # Raspberry Pi 3, 4, Zero 2 (32-bit)
            add_compile_options(-mfp16-format=ieee -mno-unaligned-access)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (BLOOM_AVX512)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX512>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if (BLOOM_AVX512_VBMI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VBMI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VBMI__>)
            endif()
            if (BLOOM_AVX512_VNNI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
            endif()
        elseif (BLOOM_AVX2)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        elseif (BLOOM_AVX)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
        endif()
    else()
        if (BLOOM_F16C)
            add_compile_options(-mf16c)
        endif()
        if (BLOOM_FMA)
            add_compile_options(-mfma)
        endif()
        if (BLOOM_AVX)
            add_compile_options(-mavx)
        endif()
        if (BLOOM_AVX2)
            add_compile_options(-mavx2)
        endif()
        if (BLOOM_AVX512)
            add_compile_options(-mavx512f)
            add_compile_options(-mavx512bw)
        endif()
        if (BLOOM_AVX512_VBMI)
            add_compile_options(-mavx512vbmi)
        endif()
        if (BLOOM_AVX512_VNNI)
            add_compile_options(-mavx512vnni)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PowerPC detected")
    add_compile_options(-mcpu=native -mtune=native)
    #TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
else()
    message(STATUS "Unknown architecture")
endif()

#
# Build libraries
#

add_library(ggml OBJECT
            ggml.c
            ggml.h)

target_include_directories(ggml PUBLIC .)
target_compile_features(ggml PUBLIC c_std_11) # don't bump
target_link_libraries(ggml PUBLIC Threads::Threads)

if (BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(bloom
            bloom.cpp
            utils.cpp)

target_include_directories(bloom PUBLIC .)
target_compile_features(bloom PUBLIC cxx_std_11) # don't bump
target_link_libraries(bloom PRIVATE ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(bloom PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(bloom PRIVATE)
endif()

add_executable(quantize-bloom
               quantize.cpp)
target_link_libraries(quantize-bloom PRIVATE bloom)
target_compile_features(quantize-bloom PRIVATE cxx_std_11)
