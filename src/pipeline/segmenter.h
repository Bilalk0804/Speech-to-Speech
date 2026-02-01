#ifndef SEGMENTER_H
#define SEGMENTER_H

#include <cstddef>

class Segmenter {
public:
    Segmenter();
    ~Segmenter();
    
    // Voice Activity Detection
    bool process(const float* audio, size_t frames);
    bool isSpeechActive() const;
    
    // Audio chunking
    size_t getChunkSize() const;
    
private:
    bool speech_active;
    float energy_threshold;
};

#endif // SEGMENTER_H
