cmake_minimum_required(VERSION 3.12...3.22)

project("bloom.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_STANDALONE ON)

    # configure project version
    # TODO
else()
    set(LLAMA_STANDALONE OFF)
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(LLAMA_WASM_SINGLE_FILE "llama: embed WASM inside the generated llama.js" ON)
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


#
# Build libraries
#

add_library(ggml OBJECT
            ggml.c
            ggml.h)

target_include_directories(ggml PUBLIC .)
target_compile_features(ggml PUBLIC c_std_11) # don't bump
# target_link_libraries(ggml PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})

if (BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(bloom
            bloom.cpp
            utils.cpp)

target_include_directories(bloom PUBLIC .)
target_compile_features(bloom PUBLIC cxx_std_11) # don't bump
target_link_libraries(bloom PRIVATE ggml ${LLAMA_EXTRA_LIBS})

if (BUILD_SHARED_LIBS)
    set_target_properties(bloom PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(bloom PRIVATE)
endif()

add_executable(quantize-bloom
               quantize.cpp)
target_link_libraries(quantize-bloom PRIVATE bloom)
target_compile_features(quantize-bloom PRIVATE cxx_std_11)
