cmake_minimum_required(VERSION 3.18)

# Create main s2s library
add_library(s2s_core
    # Core pipeline
    pipeline/pipeline.cpp
    pipeline/pipeline.h
    
    # ASR components
    asr/asr.cpp
    asr/asr.h
    
    # Machine Translation
    mt/mt.cpp
    mt/mt.h
    
    # Utilities
    utils/matrix_ops.cpp
    utils/model_converter.cpp
    utils/model_loader.cpp
    
    # Audio backend
    audio/audio_alsa.cpp
    audio/audio_jack.cpp
    
    # Kernel implementations
    asr/kernels/ref/ref_kernels.c
)

# Add NEON kernels if enabled
if(ENABLE_NEON)
    target_sources(s2s_core PRIVATE asr/kernels/neon/simd_kernels.S)
endif()

# Add SME2 kernels if enabled
if(ENABLE_SME2)
    target_sources(s2s_core PRIVATE asr/kernels/sme2/matrix_gemm.S)
endif()

# Set include directories
target_include_directories(s2s_core 
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

# Compile features
target_compile_features(s2s_core PUBLIC cxx_std_17)

# Link math library
target_link_libraries(s2s_core PRIVATE m)

# Audio backend libraries (optional)
find_package(ALSA QUIET)
if(ALSA_FOUND)
    target_link_libraries(s2s_core PRIVATE ${ALSA_LIBRARIES})
    message(STATUS "ALSA audio backend available")
else()
    message(STATUS "ALSA not found, ALSA backend disabled")
endif()

# Create shared library
add_library(s2s_shared SHARED
    $<TARGET_OBJECTS:s2s_core>
)

# Main executable (optional example)
add_executable(s2s_app 
    app/main.cpp
)

target_link_libraries(s2s_app PRIVATE s2s_core)
target_link_libraries(s2s_app PRIVATE m)

# Optimization flags
target_compile_options(s2s_core PRIVATE -O3 -march=native)
target_compile_options(s2s_app PRIVATE -O3 -march=native)

message(STATUS "S2S Core library created")
message(STATUS "  Target: s2s_core (static)")
message(STATUS "  Target: s2s_shared (shared)")
message(STATUS "  Target: s2s_app (executable)")
