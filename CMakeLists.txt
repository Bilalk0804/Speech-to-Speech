cmake_minimum_required(VERSION 3.18)
project(s2s_ondevice LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3")

# Detect ARM architecture
execute_process(COMMAND gcc -dumpmachine OUTPUT_VARIABLE MACHINE OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Target architecture: ${MACHINE}")

# Feature detection
option(ENABLE_NEON "Enable ARM NEON optimizations" ON)
option(ENABLE_SME2 "Enable ARM SME2 optimizations" OFF)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

# NEON support
if(ENABLE_NEON)
    message(STATUS "NEON support enabled")
    add_compile_definitions(ENABLE_NEON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -mfloat-abi=hard")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon -mfloat-abi=hard")
endif()

# SME2 support
if(ENABLE_SME2)
    message(STATUS "SME2 support enabled")
    add_compile_definitions(ENABLE_SME2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv9-a+sme2")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv9-a+sme2")
    enable_language(ASM)
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -march=armv9-a+sme2")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

# Main subdirectories
add_subdirectory(src)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Summary
message(STATUS "")
message(STATUS "Build Configuration:")
message(STATUS "  NEON Enabled: ${ENABLE_NEON}")
message(STATUS "  SME2 Enabled: ${ENABLE_SME2}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "")
