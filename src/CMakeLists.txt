add_library(s2s_core
    app/main.cpp
    audio/audio_alsa.cpp
    pipeline/pipeline.cpp
    # other files...
)

target_include_directories(s2s_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(s2s_core PUBLIC cxx_std_17)

if(ENABLE_NEON)
    target_compile_definitions(s2s_core PUBLIC ENABLE_NEON)
    target_compile_options(s2s_core PRIVATE -mfpu=neon -mfloat-abi=hard)
    target_sources(s2s_core PRIVATE asr/kernels/neon/asm_kernels.S)
endif()

if(ENABLE_SME2)
    target_compile_definitions(s2s_core PUBLIC ENABLE_SME2)
    # SME2/SVE flags are target dependent; set via toolchain file or CFLAGS.
    target_sources(s2s_core PRIVATE asr/kernels/sme2/asm_kernels.S)
endif()
