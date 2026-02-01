// JACK audio backend implementation
#include "audio_backend.h"

class JACKBackend : public AudioBackend {
public:
    bool initialize() override { return true; }
    void shutdown() override {}
    int capture(float* buffer, size_t frames) override { return 0; }
    int playback(const float* buffer, size_t frames) override { return 0; }
    int sampleRate() const override { return 44100; }
    size_t bufferSize() const override { return 1024; }
};
