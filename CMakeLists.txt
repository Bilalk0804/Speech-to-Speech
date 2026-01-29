cmake_minimum_required(VERSION 3.16)
project(s2s-ondevice)

# Project configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add subdirectories
add_subdirectory(src)

# Testing
enable_testing()
add_subdirectory(tests)
