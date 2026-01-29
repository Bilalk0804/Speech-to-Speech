#ifndef MICRO_PROFILER_H
#define MICRO_PROFILER_H

#include <chrono>

class MicroProfiler {
public:
    MicroProfiler(const char* name);
    ~MicroProfiler();
    
    void start();
    void stop();
    double elapsedMs() const;
    
private:
    const char* name;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
};

#endif // MICRO_PROFILER_H
