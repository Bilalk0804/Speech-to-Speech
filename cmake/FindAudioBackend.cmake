# FindAudioBackend.cmake
# Detect and configure audio backend (ALSA, JACK, etc.)

find_package(ALSA)
find_package(JACK)

if(ALSA_FOUND)
    message(STATUS "ALSA audio backend found")
elseif(JACK_FOUND)
    message(STATUS "JACK audio backend found")
else()
    message(WARNING "No audio backend detected")
endif()
