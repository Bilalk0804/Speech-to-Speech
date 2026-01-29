cmake_minimum_required(VERSION 3.18)
project(s2s_ondevice LANGUAGES C CXX)

option(ENABLE_NEON "Enable NEON optimizations" ON)
option(ENABLE_SME2 "Enable SME2 optimizations" OFF)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)
