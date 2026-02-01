#ifndef AUDIO_BACKEND_H
#define AUDIO_BACKEND_H

#include <cstdint>
#include <cstddef>

class AudioBackend {
public:
    virtual ~AudioBackend() = default;
    
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    
    virtual int capture(float* buffer, size_t frames) = 0;
    virtual int playback(const float* buffer, size_t frames) = 0;
    
    virtual int sampleRate() const = 0;
    virtual size_t bufferSize() const = 0;
};

#endif // AUDIO_BACKEND_H
